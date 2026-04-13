// e2e/bench_e2e.cu
//
// End-to-end single-GPU MoE layer benchmark.
// Full pipeline: route → pack → expert GEMMs → unpack+combine
//
// Two modes that do identical work:
//   naive: gate_kernel → softmax_kernel → topk_kernel → count → CUB → scatter → GEMMs → unpack
//   fused: fused_route_kernel → CUB → scatter → GEMMs → unpack
//
// Expert GEMMs use cuBLAS hgemm, run sequentially (one per expert).
//
// Build (needs cuBLAS 12 from conda env to match driver 572.83):
//   CUBLAS12=$HOME/miniconda3/envs/myenv/lib/python3.9/site-packages/nvidia/cublas
//   nvcc -O3 -std=c++17 -arch=sm_86 -I${CUBLAS12}/include \
//     e2e/bench_e2e.cu -o e2e/bench_e2e \
//     -Xlinker -rpath=${CUBLAS12}/lib \
//     -Xlinker ${CUBLAS12}/lib/libcublas.so.12 \
//     -Xlinker ${CUBLAS12}/lib/libcublasLt.so.12 \
//     -lnvToolsExt
// Run:
//   ./e2e/bench_e2e                          # mini-test only
//   ./e2e/bench_e2e syn_uniform_T2048
//   ./e2e/bench_e2e syn_zipf_T8192 64 results

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <nvToolsExt.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err));                    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,      \
                         __LINE__, (int)st);                                    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// fp16 host helpers
// ---------------------------------------------------------------------------

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
               mant &= 0x3ff; f = (sign<<31)|((exp+127-15)<<23)|(mant<<13); }
    } else if (exp == 31) { f = (sign<<31)|0x7f800000|(mant<<13); }
    else { f = (sign<<31)|((exp+127-15)<<23)|(mant<<13); }
    float r; std::memcpy(&r, &f, 4); return r;
}

static uint16_t float_to_half(float val) {
    uint32_t f; std::memcpy(&f, &val, 4);
    uint32_t sign = (f>>31)&1; int32_t exp = ((f>>23)&0xff)-127; uint32_t mant = f&0x7fffff;
    if (exp > 15)  return (uint16_t)((sign<<15)|0x7c00);
    if (exp < -14) return (uint16_t)(sign<<15);
    return (uint16_t)((sign<<15)|((exp+15)<<10)|(mant>>13));
}

// ---------------------------------------------------------------------------
// .npy loader
// ---------------------------------------------------------------------------

struct NpyArray { std::vector<uint8_t> data; size_t rows=0, cols=0; std::string dtype; };

static NpyArray load_npy(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "[npy] cannot open %s\n", path.c_str()); std::exit(1); }
    char magic[6];
    if (std::fread(magic,1,6,fp)!=6||std::memcmp(magic,"\x93NUMPY",6)!=0) { std::exit(1); }
    uint8_t major=0, minor=0;
    if (std::fread(&major,1,1,fp)!=1) std::exit(1);
    if (std::fread(&minor,1,1,fp)!=1) std::exit(1);
    size_t hlen=0;
    if (major==1){uint16_t h;if(std::fread(&h,2,1,fp)!=1)std::exit(1);hlen=h;}
    else{uint32_t h;if(std::fread(&h,4,1,fp)!=1)std::exit(1);hlen=h;}
    std::string header(hlen,' ');
    if (std::fread(&header[0],1,hlen,fp)!=hlen) std::exit(1);
    NpyArray arr;
    auto dp=header.find("'descr':");auto q1=header.find('\'',dp+8);auto q2=header.find('\'',q1+1);
    arr.dtype=header.substr(q1+1,q2-q1-1);
    auto sp=header.find("'shape':");auto lp=header.find('(',sp);auto rp=header.find(')',lp);
    std::string ss=header.substr(lp+1,rp-lp-1);size_t c=ss.find(',');
    arr.rows=std::strtoul(ss.substr(0,c).c_str(),nullptr,10);
    arr.cols=std::strtoul(ss.substr(c+1).c_str(),nullptr,10);
    size_t elt=(arr.dtype=="<f2")?2:(arr.dtype=="<i4")?4:0;
    if(!elt){std::exit(1);}
    size_t nb=arr.rows*arr.cols*elt; arr.data.resize(nb);
    if(std::fread(arr.data.data(),1,nb,fp)!=nb)std::exit(1);
    std::fclose(fp);
    std::printf("[npy] %-50s  (%zu,%zu) %s\n", path.c_str(), arr.rows, arr.cols, arr.dtype.c_str());
    return arr;
}

// ---------------------------------------------------------------------------
// RNG for random weight matrices
// ---------------------------------------------------------------------------

static void fill_random_half(uint16_t* buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = float_to_half(((float)(s >> 16) / 65536.0f - 0.5f) * 0.02f);
    }
}

static void fill_random_float(float* buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = ((float)(s >> 16) / 65536.0f - 0.5f) * 0.1f;
    }
}

// ===========================================================================
// NAIVE MODE: separate routing kernels (3 launches, logits/probs hit HBM)
// ===========================================================================

// one thread per (token, expert), writes logits[T, E]
__global__ void gate_kernel(
    const __half* __restrict__ emb,     // [T, d]
    const float*  __restrict__ W_gate,  // [d, E]
    float*        __restrict__ logits,  // [T, E]
    int T, int E, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * E) return;
    int t = idx / E, e = idx % E;
    float acc = 0.0f;
    for (int i = 0; i < d; i++)
        acc += __half2float(emb[t * d + i]) * W_gate[i * E + e];
    logits[t * E + e] = acc;
}

// one thread per token, reads logits[T, E], writes probs[T, E]
__global__ void softmax_kernel(
    const float* __restrict__ logits,
    float*       __restrict__ probs,
    int T, int E)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* row = logits + t * E;
    float* out = probs + t * E;
    float mx = row[0];
    for (int e = 1; e < E; e++) mx = fmaxf(mx, row[e]);
    float sum = 0;
    for (int e = 0; e < E; e++) { out[e] = expf(row[e] - mx); sum += out[e]; }
    for (int e = 0; e < E; e++) out[e] /= sum;
}

// one thread per token, K passes of argmax
__global__ void topk_kernel(
    const float* __restrict__ probs,
    int32_t*     __restrict__ assignments,
    float*       __restrict__ weights,
    int32_t*     __restrict__ expert_count,
    int T, int K, int E)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float lp[128];  // E <= 128
    for (int e = 0; e < E; e++) lp[e] = probs[t * E + e];
    for (int k = 0; k < K; k++) {
        int best = 0;
        for (int e = 1; e < E; e++) if (lp[e] > lp[best]) best = e;
        assignments[t * K + k] = best;
        weights[t * K + k] = lp[best];
        atomicAdd(&expert_count[best], 1);
        lp[best] = -1.0f;
    }
}

// ===========================================================================
// FUSED MODE: single routing kernel
// ===========================================================================

__global__ void fused_route_kernel(
    const __half* __restrict__ emb,
    const float*  __restrict__ W_gate,
    int T, int K, int E, int d,
    int32_t* __restrict__ assignments,
    float*   __restrict__ weights,
    int32_t* __restrict__ expert_count)
{
    int t = blockIdx.x, e = threadIdx.x;
    if (t >= T || e >= E) return;

    float logit = 0.0f;
    for (int i = 0; i < d; i++)
        logit += __half2float(emb[t * d + i]) * W_gate[i * E + e];

    extern __shared__ float smem[];
    smem[e] = logit;
    __syncthreads();

    __shared__ int32_t s_idx[8];
    __shared__ float   s_val[8];
    if (e == 0) {
        float mx = smem[0];
        for (int i = 1; i < E; i++) mx = fmaxf(mx, smem[i]);
        float sum = 0;
        for (int i = 0; i < E; i++) { smem[i] = expf(smem[i] - mx); sum += smem[i]; }
        for (int i = 0; i < E; i++) smem[i] /= sum;
        for (int k = 0; k < K; k++) {
            int best = 0;
            for (int i = 1; i < E; i++) if (smem[i] > smem[best]) best = i;
            s_idx[k] = best; s_val[k] = smem[best]; smem[best] = -1.0f;
        }
    }
    __syncthreads();

    if (e < K) {
        assignments[t * K + e] = s_idx[e];
        weights[t * K + e]     = s_val[e];
        atomicAdd(&expert_count[s_idx[e]], 1);
    }
}

// ===========================================================================
// SHARED: scatter, weight scatter, unpack+combine
// ===========================================================================

__global__ void scatter_kernel(
    const __half*  __restrict__ emb,
    const int32_t* __restrict__ assignments,
    const int32_t* __restrict__ expert_start,
    int32_t*       __restrict__ cursor,
    __half*        __restrict__ packed,
    int32_t*       __restrict__ perm,
    int T, int K, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * K) return;
    int t = idx / K, e = assignments[idx];
    int pos = expert_start[e] + atomicAdd(&cursor[e], 1);
    const __half* src = emb + (size_t)t * d;
    __half* dst = packed + (size_t)pos * d;
    for (int i = 0; i < d; i++) dst[i] = src[i];
    perm[pos] = t;
}

// scatter routing weights into packed order (same atomic pattern as scatter)
__global__ void scatter_weights_kernel(
    const float*   __restrict__ route_weights,
    const int32_t* __restrict__ assignments,
    const int32_t* __restrict__ expert_start,
    int32_t*       __restrict__ cursor,
    float*         __restrict__ packed_weights,
    int T, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * K) return;
    int e = assignments[idx];
    int pos = expert_start[e] + atomicAdd(&cursor[e], 1);
    packed_weights[pos] = route_weights[idx];
}

// unpack: output[t] += weight[pos] * expert_out[pos], accumulated in float32
__global__ void unpack_combine_kernel(
    const __half*  __restrict__ expert_out,
    const float*   __restrict__ packed_weights,
    const int32_t* __restrict__ perm,
    float*         __restrict__ output,  // [T, d] zeroed float32
    int TK, int d)
{
    int pos = blockIdx.x;
    if (pos >= TK) return;
    int t = perm[pos];
    float w = packed_weights[pos];
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        atomicAdd(&output[t * d + i], w * __half2float(expert_out[pos * d + i]));
}

__global__ void f32_to_f16_kernel(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// ===========================================================================
// E2E timing result
// ===========================================================================

struct E2EResult {
    float route_ms;
    float pack_ms;    // prefix sum + scatter + weight scatter
    float gemm_ms;
    float unpack_ms;
    float total_ms;
};

// ===========================================================================
// run_e2e: full MoE layer
// ===========================================================================

static E2EResult run_e2e(
    const __half* d_emb,
    const float*  d_W_gate,
    const __half* d_W1,        // [E, d, d_ffn]
    const __half* d_W2,        // [E, d_ffn, d]
    int T, int K, int E, int d, int d_ffn,
    bool use_fused,
    cublasHandle_t cublas,
    cudaStream_t stream = 0)
{
    int TK = T * K;
    int blk = 256;

    // alloc intermediates
    float*   d_logits = nullptr;
    float*   d_probs  = nullptr;
    if (!use_fused) {
        CHECK_CUDA(cudaMalloc(&d_logits, (size_t)T * E * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_probs,  (size_t)T * E * sizeof(float)));
    }
    int32_t *d_asgn, *d_ecount, *d_estart, *d_perm, *d_cursor, *d_cursor2;
    float   *d_rwt, *d_pwt, *d_out_f32;
    __half  *d_packed, *d_expert_out, *d_output, *d_ffn_tmp;

    CHECK_CUDA(cudaMalloc(&d_asgn,       (size_t)TK * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_rwt,        (size_t)TK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ecount,     E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_estart,     E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_packed,     (size_t)TK * d * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_perm,       (size_t)TK * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_cursor,     E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_pwt,        (size_t)TK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cursor2,    E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_expert_out, (size_t)TK * d * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_f32,    (size_t)T * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output,     (size_t)T * d * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_ffn_tmp,    (size_t)TK * d_ffn * sizeof(__half)));

    void* d_cub = nullptr; size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_cub, cub_bytes, d_ecount, d_estart, E, stream);
    CHECK_CUDA(cudaMalloc(&d_cub, cub_bytes));

    cudaEvent_t ev[5];
    for (int i = 0; i < 5; i++) CHECK_CUDA(cudaEventCreate(&ev[i]));

    CHECK_CUDA(cudaMemsetAsync(d_ecount, 0, E * sizeof(int32_t), stream));

    // ===================== ROUTING =====================
    CHECK_CUDA(cudaEventRecord(ev[0], stream));

    if (use_fused) {
        nvtxRangePush("fused_route");
        fused_route_kernel<<<T, E, E * sizeof(float), stream>>>(
            d_emb, d_W_gate, T, K, E, d, d_asgn, d_rwt, d_ecount);
        nvtxRangePop();
    } else {
        nvtxRangePush("gate_kernel");
        gate_kernel<<<((T*E)+blk-1)/blk, blk, 0, stream>>>(d_emb, d_W_gate, d_logits, T, E, d);
        nvtxRangePop();
        nvtxRangePush("softmax_kernel");
        softmax_kernel<<<(T+blk-1)/blk, blk, 0, stream>>>(d_logits, d_probs, T, E);
        nvtxRangePop();
        nvtxRangePush("topk_kernel");
        topk_kernel<<<(T+blk-1)/blk, blk, 0, stream>>>(d_probs, d_asgn, d_rwt, d_ecount, T, K, E);
        nvtxRangePop();
    }
    CHECK_CUDA(cudaEventRecord(ev[1], stream));

    // ===================== PACK =====================
    nvtxRangePush("cub_prefix_sum");
    cub::DeviceScan::ExclusiveSum(d_cub, cub_bytes, d_ecount, d_estart, E, stream);
    nvtxRangePop();

    CHECK_CUDA(cudaMemsetAsync(d_cursor,  0, E * sizeof(int32_t), stream));
    CHECK_CUDA(cudaMemsetAsync(d_cursor2, 0, E * sizeof(int32_t), stream));
    int grid_tk = (TK + blk - 1) / blk;

    nvtxRangePush("scatter");
    scatter_kernel<<<grid_tk, blk, 0, stream>>>(
        d_emb, d_asgn, d_estart, d_cursor, d_packed, d_perm, T, K, d);
    nvtxRangePop();

    nvtxRangePush("scatter_weights");
    scatter_weights_kernel<<<grid_tk, blk, 0, stream>>>(
        d_rwt, d_asgn, d_estart, d_cursor2, d_pwt, T, K);
    nvtxRangePop();

    CHECK_CUDA(cudaEventRecord(ev[2], stream));

    // need expert_start/count on host for the GEMM loop
    std::vector<int32_t> h_estart(E), h_ecount(E);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(h_estart.data(), d_estart, E*sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ecount.data(), d_ecount, E*sizeof(int32_t), cudaMemcpyDeviceToHost));

    // ===================== EXPERT GEMMs =====================
    // Each expert: up-proj [count_e, d] @ [d, d_ffn] then down-proj [count_e, d_ffn] @ [d_ffn, d]
    // Row-major trick: cublas sees col-major, so we pass row-major A[M,K] as col-major [K,M]
    //   C[M,N] = A[M,K] @ B[K,N]  becomes  cublas(N, M, K, B, N, A, K, C, N) with OP_N, OP_N
    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    CHECK_CUBLAS(cublasSetStream(cublas, stream));

    CHECK_CUDA(cudaEventRecord(ev[3], stream));
    nvtxRangePush("expert_gemms");
    for (int e = 0; e < E; e++) {
        int cnt = h_ecount[e];
        if (cnt == 0) continue;
        int off = h_estart[e];

        const __half* A  = d_packed   + (size_t)off * d;
        const __half* B1 = d_W1       + (size_t)e * d * d_ffn;
        const __half* B2 = d_W2       + (size_t)e * d_ffn * d;
        __half* mid      = d_ffn_tmp  + (size_t)off * d_ffn;
        __half* out      = d_expert_out + (size_t)off * d;

        // up-proj: [cnt, d] @ [d, d_ffn] → [cnt, d_ffn]
        CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            d_ffn, cnt, d, &alpha_h, B1, d_ffn, A, d, &beta_h, mid, d_ffn));

        // down-proj: [cnt, d_ffn] @ [d_ffn, d] → [cnt, d]
        CHECK_CUBLAS(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            d, cnt, d_ffn, &alpha_h, B2, d, mid, d_ffn, &beta_h, out, d));
    }
    nvtxRangePop();

    // ===================== UNPACK + COMBINE =====================
    CHECK_CUDA(cudaMemsetAsync(d_out_f32, 0, (size_t)T * d * sizeof(float), stream));

    nvtxRangePush("unpack_combine");
    unpack_combine_kernel<<<TK, blk, 0, stream>>>(
        d_expert_out, d_pwt, d_perm, d_out_f32, TK, d);
    nvtxRangePop();

    nvtxRangePush("f32_to_f16");
    f32_to_f16_kernel<<<(T*d+blk-1)/blk, blk, 0, stream>>>(d_out_f32, d_output, T*d);
    nvtxRangePop();

    CHECK_CUDA(cudaEventRecord(ev[4], stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    E2EResult res;
    CHECK_CUDA(cudaEventElapsedTime(&res.route_ms,  ev[0], ev[1]));
    CHECK_CUDA(cudaEventElapsedTime(&res.pack_ms,   ev[1], ev[2]));
    CHECK_CUDA(cudaEventElapsedTime(&res.gemm_ms,   ev[3], ev[4]));
    // unpack is inside the gemm event range — let me fix the event placement
    // Actually ev[3] is after the D2H copy, and ev[4] is after unpack.
    // gemm + unpack are between ev[3] and ev[4].
    // Let me just report gemm+unpack together and note it.
    // For now, total = route + pack + (gemm+unpack)
    float gemm_unpack;
    CHECK_CUDA(cudaEventElapsedTime(&gemm_unpack, ev[3], ev[4]));
    res.gemm_ms = gemm_unpack;  // includes unpack, labeled in output
    res.unpack_ms = 0;  // folded into gemm_ms for now
    res.total_ms = res.route_ms + res.pack_ms + gemm_unpack;

    // cleanup
    if (d_logits) CHECK_CUDA(cudaFree(d_logits));
    if (d_probs)  CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaFree(d_asgn));    CHECK_CUDA(cudaFree(d_rwt));
    CHECK_CUDA(cudaFree(d_ecount));  CHECK_CUDA(cudaFree(d_estart));
    CHECK_CUDA(cudaFree(d_packed));  CHECK_CUDA(cudaFree(d_perm));
    CHECK_CUDA(cudaFree(d_cursor));  CHECK_CUDA(cudaFree(d_pwt));
    CHECK_CUDA(cudaFree(d_cursor2)); CHECK_CUDA(cudaFree(d_expert_out));
    CHECK_CUDA(cudaFree(d_out_f32)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_ffn_tmp)); CHECK_CUDA(cudaFree(d_cub));
    for (int i = 0; i < 5; i++) CHECK_CUDA(cudaEventDestroy(ev[i]));

    return res;
}

// ===========================================================================
// Mini-test
// ===========================================================================

static bool run_mini_test() {
    std::printf("\n=== Mini-test (T=16, E=4, K=2, d=8, d_ffn=32) ===\n\n");

    const int T=16, E=4, K=2, d=8, d_ffn=32, TK=T*K;

    std::vector<uint16_t> h_emb(T*d);
    fill_random_half(h_emb.data(), T*d, 1);
    std::vector<float> h_gate(d*E);
    fill_random_float(h_gate.data(), d*E, 2);
    std::vector<uint16_t> h_W1((size_t)E*d*d_ffn), h_W2((size_t)E*d_ffn*d);
    fill_random_half(h_W1.data(), E*d*d_ffn, 3);
    fill_random_half(h_W2.data(), E*d_ffn*d, 4);

    __half *d_emb, *d_W1, *d_W2; float *d_gate;
    CHECK_CUDA(cudaMalloc(&d_emb,  T*d*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_gate, d*E*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W1,   (size_t)E*d*d_ffn*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_W2,   (size_t)E*d_ffn*d*sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_emb,  h_emb.data(),  T*d*2,                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(),  d*E*sizeof(float),       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1,   h_W1.data(),    (size_t)E*d*d_ffn*2,    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2,   h_W2.data(),    (size_t)E*d_ffn*d*2,    cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    for (int mode = 0; mode < 2; mode++) {
        bool fused = (mode == 1);
        auto r = run_e2e(d_emb, d_gate, d_W1, d_W2, T, K, E, d, d_ffn, fused, cublas);
        std::printf("  %-6s: route=%.3f  pack=%.3f  gemm+unpack=%.3f  total=%.3f ms\n",
                    fused ? "fused" : "naive", r.route_ms, r.pack_ms, r.gemm_ms, r.total_ms);
    }

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDA(cudaFree(d_emb)); CHECK_CUDA(cudaFree(d_gate));
    CHECK_CUDA(cudaFree(d_W1));  CHECK_CUDA(cudaFree(d_W2));
    std::printf("\n  Mini-test done.\n");
    return true;
}

// ===========================================================================
// Full benchmark
// ===========================================================================

static void run_full(const std::string& prefix, int E_cli, const std::string& data_dir) {
    std::printf("\n=== E2E MoE Layer Benchmark ===\n\n");

    NpyArray emb_npy  = load_npy(data_dir + "/" + prefix + "_embeddings.npy");
    NpyArray asgn_npy = load_npy(data_dir + "/" + prefix + "_assignments.npy");

    int T = (int)emb_npy.rows, d = (int)emb_npy.cols;
    int K = (int)asgn_npy.cols, E = E_cli;
    int d_ffn = d * 4;
    int TK = T * K;

    std::printf("T=%d  K=%d  E=%d  d=%d  d_ffn=%d  packed_rows=%d\n", T, K, E, d, d_ffn, TK);

    // check if expert weights will fit in GPU memory
    size_t w1_bytes = (size_t)E * d * d_ffn * 2;
    size_t w2_bytes = (size_t)E * d_ffn * d * 2;
    size_t ffn_tmp  = (size_t)TK * d_ffn * 2;
    size_t packed_bytes = (size_t)TK * d * 2;
    size_t total_est = w1_bytes + w2_bytes + ffn_tmp + packed_bytes * 2 + (size_t)T*d*4;
    std::printf("Estimated GPU memory: %.1f MB\n", total_est / 1e6);

    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    std::printf("GPU memory: %.1f MB free / %.1f MB total\n\n", free_mem/1e6, total_mem/1e6);

    if (total_est > free_mem * 0.9) {
        std::printf("WARNING: estimated memory exceeds 90%% of free GPU memory.\n");
        std::printf("Consider using a smaller d_ffn or T.\n\n");
    }

    // generate random weights
    std::printf("Generating random expert weights...\n");
    std::vector<float> h_gate(d * E);
    fill_random_float(h_gate.data(), d * E, 42);

    size_t w1n = (size_t)E * d * d_ffn, w2n = (size_t)E * d_ffn * d;
    std::printf("  W1: [%d, %d, %d] = %.1f MB\n", E, d, d_ffn, w1n*2.0/1e6);
    std::printf("  W2: [%d, %d, %d] = %.1f MB\n", E, d_ffn, d, w2n*2.0/1e6);

    std::vector<uint16_t> h_W1(w1n), h_W2(w2n);
    fill_random_half(h_W1.data(), w1n, 100);
    fill_random_half(h_W2.data(), w2n, 200);

    // upload
    __half *d_emb, *d_W1, *d_W2; float *d_gate;
    CHECK_CUDA(cudaMalloc(&d_emb,  (size_t)T*d*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_gate, (size_t)d*E*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W1,   w1n*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_W2,   w2n*sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_emb,  emb_npy.data.data(), (size_t)T*d*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(), d*E*sizeof(float),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1,   h_W1.data(), w1n*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2,   h_W2.data(), w2n*2, cudaMemcpyHostToDevice));
    h_W1.clear(); h_W1.shrink_to_fit();
    h_W2.clear(); h_W2.shrink_to_fit();

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    // warmup
    for (int m = 0; m < 2; m++)
        run_e2e(d_emb, d_gate, d_W1, d_W2, T, K, E, d, d_ffn, m==1, cublas);

    // benchmark
    const int NRUNS = 5;
    for (int m = 0; m < 2; m++) {
        bool fused = (m == 1);
        E2EResult tot = {};
        for (int i = 0; i < NRUNS; i++) {
            auto r = run_e2e(d_emb, d_gate, d_W1, d_W2, T, K, E, d, d_ffn, fused, cublas);
            tot.route_ms += r.route_ms;
            tot.pack_ms  += r.pack_ms;
            tot.gemm_ms  += r.gemm_ms;
            tot.total_ms += r.total_ms;
        }
        float ar = tot.route_ms/NRUNS, ap = tot.pack_ms/NRUNS;
        float ag = tot.gemm_ms/NRUNS,  at = tot.total_ms/NRUNS;
        float pack_frac = (ar + ap) / at * 100;

        std::printf("[%s] avg over %d runs:\n", fused ? "FUSED" : "NAIVE", NRUNS);
        std::printf("  route:         %8.3f ms  (%5.1f%%)\n", ar, ar/at*100);
        std::printf("  pack:          %8.3f ms  (%5.1f%%)\n", ap, ap/at*100);
        std::printf("  gemm+unpack:   %8.3f ms  (%5.1f%%)\n", ag, ag/at*100);
        std::printf("  ────────────────────────────\n");
        std::printf("  total:         %8.3f ms\n", at);
        std::printf("  packing frac:  %5.1f%%  (route+pack)\n\n", pack_frac);
    }

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDA(cudaFree(d_emb)); CHECK_CUDA(cudaFree(d_gate));
    CHECK_CUDA(cudaFree(d_W1));  CHECK_CUDA(cudaFree(d_W2));
}

int main(int argc, char** argv) {
    if (!run_mini_test()) return 1;
    if (argc > 1) {
        std::string prefix = argv[1];
        int E_cli = (argc > 2) ? std::atoi(argv[2]) : 64;
        std::string data_dir = (argc > 3) ? argv[3] : "results";
        run_full(prefix, E_cli, data_dir);
    }
    return 0;
}
