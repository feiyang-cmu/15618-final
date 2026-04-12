// fused/fused_routing_pack.cu
//
// Fused gate → softmax → topK → scatter kernel for MoE token packing.
// The routing decision stays in shared memory, avoiding the logits[T,E]
// and probs[T,E] HBM round-trips that a separate-kernel approach needs.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 fused/fused_routing_pack.cu -o fused/fused_routing_pack -lnvToolsExt
// Run:
//   ./fused/fused_routing_pack                        # mini-tests only
//   ./fused/fused_routing_pack syn_uniform_T2048      # full run

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

// ---------------------------------------------------------------------------
// fp16 host-side helpers (no __half arithmetic on host before CUDA 12)
// ---------------------------------------------------------------------------

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { // subnormal
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

static uint16_t float_to_half(float val) {
    uint32_t f;
    std::memcpy(&f, &val, 4);
    uint32_t sign = (f >> 31) & 1;
    int32_t  exp  = ((f >> 23) & 0xff) - 127;
    uint32_t mant = f & 0x7fffff;
    if (exp > 15)  return (uint16_t)((sign << 15) | 0x7c00);
    if (exp < -14) return (uint16_t)(sign << 15);
    return (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
}

// ---------------------------------------------------------------------------
// .npy loader (same as naive_gpu.cu — duplicated to keep files standalone)
// ---------------------------------------------------------------------------

struct NpyArray {
    std::vector<uint8_t> data;
    size_t rows = 0, cols = 0;
    std::string dtype;
};

static NpyArray load_npy(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "[npy] cannot open %s\n", path.c_str()); std::exit(1); }
    char magic[6];
    if (std::fread(magic, 1, 6, fp) != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        { std::fprintf(stderr, "[npy] bad magic\n"); std::exit(1); }
    uint8_t major = 0, minor = 0;
    if (std::fread(&major, 1, 1, fp) != 1) std::exit(1);
    if (std::fread(&minor, 1, 1, fp) != 1) std::exit(1);
    size_t header_len = 0;
    if (major == 1) { uint16_t h; if (std::fread(&h, 2, 1, fp) != 1) std::exit(1); header_len = h; }
    else            { uint32_t h; if (std::fread(&h, 4, 1, fp) != 1) std::exit(1); header_len = h; }
    std::string header(header_len, ' ');
    if (std::fread(&header[0], 1, header_len, fp) != header_len) std::exit(1);

    NpyArray arr;
    auto d_pos = header.find("'descr':"); auto q1 = header.find('\'', d_pos+8); auto q2 = header.find('\'', q1+1);
    arr.dtype = header.substr(q1+1, q2-q1-1);
    auto s_pos = header.find("'shape':"); auto lp = header.find('(', s_pos); auto rp = header.find(')', lp);
    std::string ss = header.substr(lp+1, rp-lp-1); size_t comma = ss.find(',');
    arr.rows = std::strtoul(ss.substr(0, comma).c_str(), nullptr, 10);
    arr.cols = std::strtoul(ss.substr(comma+1).c_str(), nullptr, 10);
    size_t elt = (arr.dtype == "<f2") ? 2 : (arr.dtype == "<i4") ? 4 : 0;
    if (!elt) { std::fprintf(stderr, "[npy] unsupported dtype\n"); std::exit(1); }
    size_t nbytes = arr.rows * arr.cols * elt;
    arr.data.resize(nbytes);
    if (std::fread(arr.data.data(), 1, nbytes, fp) != nbytes) std::exit(1);
    std::fclose(fp);
    std::printf("[npy] %-50s  (%zu, %zu)  %s\n", path.c_str(), arr.rows, arr.cols, arr.dtype.c_str());
    return arr;
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

// gate GEMV → softmax → topK, serial. for verification only.
static void cpu_route(
    const uint16_t* emb, const float* W_gate,
    int T, int K, int E, int d,
    int32_t* assignments, float* weights)
{
    std::vector<float> logits(E);
    for (int t = 0; t < T; t++) {
        for (int e = 0; e < E; e++) {
            float acc = 0;
            for (int i = 0; i < d; i++)
                acc += half_to_float(emb[t * d + i]) * W_gate[i * E + e];
            logits[e] = acc;
        }
        // softmax
        float mx = *std::max_element(logits.begin(), logits.end());
        float sum = 0;
        for (int e = 0; e < E; e++) { logits[e] = std::exp(logits[e] - mx); sum += logits[e]; }
        for (int e = 0; e < E; e++) logits[e] /= sum;

        // topK: K passes of argmax
        for (int k = 0; k < K; k++) {
            int best = 0;
            for (int e = 1; e < E; e++)
                if (logits[e] > logits[best]) best = e;
            assignments[t * K + k] = best;
            weights[t * K + k] = logits[best];
            logits[best] = -1.0f;
        }
    }
}

// serial scatter (same as naive_gpu.cu and sequential_cpu.cpp)
static void cpu_scatter(
    const uint16_t* emb, const int32_t* asgn,
    int T, int K, int E, int d,
    uint16_t* packed, int32_t* expert_start, int32_t* expert_count, int32_t* perm)
{
    std::memset(expert_count, 0, E * sizeof(int32_t));
    for (int t = 0; t < T; t++)
        for (int k = 0; k < K; k++)
            expert_count[asgn[t * K + k]]++;
    int32_t run = 0;
    for (int e = 0; e < E; e++) { expert_start[e] = run; run += expert_count[e]; }
    std::vector<int32_t> cursor(E, 0);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            int e = asgn[t * K + k];
            int pos = expert_start[e] + cursor[e]++;
            std::memcpy(packed + (size_t)pos * d, emb + (size_t)t * d, d * sizeof(uint16_t));
            perm[pos] = t;
        }
    }
}

// ===========================================================================
// Fused routing kernel: gate GEMV → softmax → topK → count
// ===========================================================================
// blockDim.x = E, gridDim.x = T. One thread per expert, one block per token.
//
// Why E threads: the GEMV has one dot product per expert, and E is small (64).
// Gate weight reads are coalesced (adjacent threads = adjacent columns).
// Embedding reads broadcast through L1 (all threads read same addr).
//
// After GEMV, thread 0 does softmax + topK serially. E=64 means ~200 flops,
// dominated by the GEMV anyway (d=2048 * E=64 = 131K flops).

__global__ void fused_route_kernel(
    const __half*  __restrict__ embeddings,    // [T, d]
    const float*   __restrict__ W_gate,        // [d, E] row-major
    int T, int K, int E, int d,
    int32_t*       __restrict__ assignments,   // [T, K] output
    float*         __restrict__ weights,        // [T, K] output
    int32_t*       __restrict__ expert_count)   // [E] output (atomicAdd, pre-zeroed)
{
    int t = blockIdx.x;
    int e = threadIdx.x;
    if (t >= T || e >= E) return;

    // GEMV: each thread computes its expert's logit
    float logit = 0.0f;
    for (int i = 0; i < d; i++)
        logit += __half2float(embeddings[t * d + i]) * W_gate[i * E + e];

    // store in shared mem for softmax + topK
    extern __shared__ float smem[];
    float* s_logits = smem;       // [E]
    s_logits[e] = logit;
    __syncthreads();

    // softmax + topK — thread 0 only, E is tiny
    __shared__ int32_t s_topk_idx[8];   // supports K <= 8
    __shared__ float   s_topk_val[8];

    if (e == 0) {
        float mx = s_logits[0];
        for (int i = 1; i < E; i++) mx = fmaxf(mx, s_logits[i]);
        float sum = 0;
        for (int i = 0; i < E; i++) { s_logits[i] = expf(s_logits[i] - mx); sum += s_logits[i]; }
        for (int i = 0; i < E; i++) s_logits[i] /= sum;

        // K passes of argmax
        for (int k = 0; k < K; k++) {
            int best = 0;
            for (int i = 1; i < E; i++)
                if (s_logits[i] > s_logits[best]) best = i;
            s_topk_idx[k] = best;
            s_topk_val[k] = s_logits[best];
            s_logits[best] = -1.0f;
        }
    }
    __syncthreads();

    // write the small outputs to global memory
    if (e < K) {
        assignments[t * K + e] = s_topk_idx[e];
        weights[t * K + e]     = s_topk_val[e];
        atomicAdd(&expert_count[s_topk_idx[e]], 1);
    }
}

// ===========================================================================
// Scatter kernel (same as naive baseline, just uses __half* instead of uint16_t*)
// ===========================================================================

__global__ void scatter_kernel(
    const __half*   __restrict__ embeddings,
    const int32_t*  __restrict__ assignments,
    const int32_t*  __restrict__ expert_start,
    int32_t*        __restrict__ cursor,
    __half*         __restrict__ packed,
    int32_t*        __restrict__ permutation,
    int T, int K, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int TK = T * K;
    if (idx >= TK) return;

    int t = idx / K;
    int e = assignments[idx];
    int slot = atomicAdd(&cursor[e], 1);
    int pos = expert_start[e] + slot;

    const __half* src = embeddings + (size_t)t * d;
    __half*       dst = packed     + (size_t)pos * d;
    for (int i = 0; i < d; i++)
        dst[i] = src[i];

    permutation[pos] = t;
}

// ===========================================================================
// Fused pipeline: route → CUB prefix sum → scatter
// ===========================================================================

struct FusedResult {
    float route_ms;
    float prefix_ms;
    float scatter_ms;
};

static FusedResult fused_pipeline(
    const __half* d_emb, const float* d_W_gate,
    int T, int K, int E, int d,
    __half*   d_packed,
    int32_t*  d_expert_start,
    int32_t*  d_expert_count,
    int32_t*  d_assignments,
    float*    d_weights,
    int32_t*  d_permutation,
    cudaStream_t stream = 0)
{
    int TK = T * K;

    int32_t* d_cursor;
    CHECK_CUDA(cudaMalloc(&d_cursor, E * sizeof(int32_t)));

    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_cub_tmp, cub_bytes, d_expert_count, d_expert_start, E, stream);
    CHECK_CUDA(cudaMalloc(&d_cub_tmp, cub_bytes));

    cudaEvent_t t0, t1, t2, t3;
    CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventCreate(&t2)); CHECK_CUDA(cudaEventCreate(&t3));

    // step 1: fused routing + count
    CHECK_CUDA(cudaMemsetAsync(d_expert_count, 0, E * sizeof(int32_t), stream));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    nvtxRangePush("fused_route");
    size_t smem = E * sizeof(float);   // s_logits
    fused_route_kernel<<<T, E, smem, stream>>>(
        d_emb, d_W_gate, T, K, E, d, d_assignments, d_weights, d_expert_count);
    nvtxRangePop();
    CHECK_CUDA(cudaEventRecord(t1, stream));

    // step 2: prefix sum
    nvtxRangePush("cub_prefix_sum");
    cub::DeviceScan::ExclusiveSum(d_cub_tmp, cub_bytes, d_expert_count, d_expert_start, E, stream);
    nvtxRangePop();
    CHECK_CUDA(cudaEventRecord(t2, stream));

    // step 3: scatter
    CHECK_CUDA(cudaMemsetAsync(d_cursor, 0, E * sizeof(int32_t), stream));
    nvtxRangePush("scatter");
    int block = 256, grid = (TK + block - 1) / block;
    scatter_kernel<<<grid, block, 0, stream>>>(
        d_emb, d_assignments, d_expert_start, d_cursor,
        d_packed, d_permutation, T, K, d);
    nvtxRangePop();
    CHECK_CUDA(cudaEventRecord(t3, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    FusedResult res;
    CHECK_CUDA(cudaEventElapsedTime(&res.route_ms,   t0, t1));
    CHECK_CUDA(cudaEventElapsedTime(&res.prefix_ms,  t1, t2));
    CHECK_CUDA(cudaEventElapsedTime(&res.scatter_ms, t2, t3));

    CHECK_CUDA(cudaFree(d_cursor));
    CHECK_CUDA(cudaFree(d_cub_tmp));
    CHECK_CUDA(cudaEventDestroy(t0)); CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaEventDestroy(t2)); CHECK_CUDA(cudaEventDestroy(t3));
    return res;
}

// ===========================================================================
// Mini-tests
// ===========================================================================

// generate a small deterministic gate weight matrix
static std::vector<float> make_test_gate(int d, int E, int seed = 7) {
    std::vector<float> W(d * E);
    // simple LCG — not great randomness but fine for a test
    uint32_t state = seed;
    for (auto& v : W) {
        state = state * 1664525u + 1013904223u;
        v = ((float)(state >> 16) / 65536.0f - 0.5f) * 0.1f;
    }
    return W;
}

static bool run_mini_tests() {
    std::printf("\n=== Mini-tests (T=16, E=4, K=2, d=8) ===\n\n");

    const int T = 16, E = 4, K = 2, d = 8;
    const int TK = T * K;

    // build test embeddings (valid fp16 values this time, since we do arithmetic)
    std::vector<uint16_t> h_emb_u16(T * d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            h_emb_u16[t * d + i] = float_to_half((float)(t * d + i) * 0.01f);

    auto h_gate = make_test_gate(d, E);

    // --- CPU reference routing ---
    std::vector<int32_t> cpu_asgn(TK);
    std::vector<float>   cpu_wt(TK);
    cpu_route(h_emb_u16.data(), h_gate.data(), T, K, E, d,
              cpu_asgn.data(), cpu_wt.data());

    // --- test 1: fused_route_kernel vs CPU ---
    {
        __half*  d_emb;
        float*   d_gate;
        int32_t* d_asgn;
        float*   d_wt;
        int32_t* d_ecount;

        CHECK_CUDA(cudaMalloc(&d_emb,    T * d * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_gate,   d * E * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_asgn,   TK * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&d_wt,     TK * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ecount, E * sizeof(int32_t)));

        CHECK_CUDA(cudaMemcpy(d_emb,  h_emb_u16.data(), T * d * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(),     d * E * sizeof(float),    cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_ecount, 0, E * sizeof(int32_t)));

        size_t smem = E * sizeof(float);
        fused_route_kernel<<<T, E, smem>>>(d_emb, d_gate, T, K, E, d, d_asgn, d_wt, d_ecount);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int32_t> gpu_asgn(TK);
        std::vector<float>   gpu_wt(TK);
        std::vector<int32_t> gpu_ecount(E);
        CHECK_CUDA(cudaMemcpy(gpu_asgn.data(),   d_asgn,   TK * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_wt.data(),     d_wt,     TK * sizeof(float),   cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_ecount.data(), d_ecount, E * sizeof(int32_t),  cudaMemcpyDeviceToHost));

        // check assignments match
        bool asgn_ok = (gpu_asgn == cpu_asgn);
        // check weights are close (fp32 accumulation can differ slightly)
        bool wt_ok = true;
        for (int i = 0; i < TK; i++) {
            if (std::fabs(gpu_wt[i] - cpu_wt[i]) > 1e-4f) {
                std::printf("  [err] weight[%d]: gpu=%.6f cpu=%.6f\n", i, gpu_wt[i], cpu_wt[i]);
                wt_ok = false;
                break;
            }
        }
        // check expert counts
        std::vector<int32_t> cpu_ecount(E, 0);
        for (auto e : cpu_asgn) cpu_ecount[e]++;
        bool ec_ok = (gpu_ecount == cpu_ecount);

        std::printf("  fused_route_kernel:\n");
        std::printf("    assignments:    %s\n", asgn_ok ? "PASS" : "FAIL");
        std::printf("    weights:        %s\n", wt_ok   ? "PASS" : "FAIL");
        std::printf("    expert_count:   %s\n", ec_ok   ? "PASS" : "FAIL");

        if (!asgn_ok) {
            for (int t = 0; t < T; t++)
                std::printf("    token %2d: gpu=[%d,%d] cpu=[%d,%d]\n", t,
                            gpu_asgn[t*K], gpu_asgn[t*K+1], cpu_asgn[t*K], cpu_asgn[t*K+1]);
        }

        CHECK_CUDA(cudaFree(d_emb));  CHECK_CUDA(cudaFree(d_gate));
        CHECK_CUDA(cudaFree(d_asgn)); CHECK_CUDA(cudaFree(d_wt));
        CHECK_CUDA(cudaFree(d_ecount));

        if (!asgn_ok || !wt_ok || !ec_ok) return false;
    }

    // --- test 2: full pipeline (route → prefix sum → scatter) ---
    {
        __half*  d_emb;
        float*   d_gate;
        __half*  d_packed;
        int32_t* d_asgn;
        float*   d_wt;
        int32_t* d_estart;
        int32_t* d_ecount;
        int32_t* d_perm;

        CHECK_CUDA(cudaMalloc(&d_emb,    T * d * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_gate,   d * E * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_packed, TK * d * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_asgn,   TK * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&d_wt,     TK * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_estart, E * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&d_ecount, E * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&d_perm,   TK * sizeof(int32_t)));

        CHECK_CUDA(cudaMemcpy(d_emb,  h_emb_u16.data(), T * d * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(),     d * E * sizeof(float),    cudaMemcpyHostToDevice));

        auto res = fused_pipeline(d_emb, d_gate, T, K, E, d,
                                  d_packed, d_estart, d_ecount, d_asgn, d_wt, d_perm);

        // copy back
        std::vector<uint16_t> gpu_packed(TK * d);
        std::vector<int32_t>  gpu_perm(TK), gpu_estart(E), gpu_ecount(E), gpu_asgn(TK);
        CHECK_CUDA(cudaMemcpy(gpu_packed.data(), d_packed, TK * d * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_perm.data(),   d_perm,   TK * sizeof(int32_t),     cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_estart.data(), d_estart, E * sizeof(int32_t),       cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_ecount.data(), d_ecount, E * sizeof(int32_t),       cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_asgn.data(),   d_asgn,   TK * sizeof(int32_t),     cudaMemcpyDeviceToHost));

        // run CPU scatter using the GPU's routing decisions (since they match)
        std::vector<uint16_t> cpu_packed(TK * d);
        std::vector<int32_t>  cpu_estart(E), cpu_ecount(E), cpu_perm(TK);
        cpu_scatter(h_emb_u16.data(), gpu_asgn.data(), T, K, E, d,
                    cpu_packed.data(), cpu_estart.data(), cpu_ecount.data(), cpu_perm.data());

        // verify: expert regions + packed contents (order-independent within regions)
        bool ok = true;
        for (int e = 0; e < E; e++) {
            if (gpu_ecount[e] != cpu_ecount[e] || gpu_estart[e] != cpu_estart[e]) {
                std::printf("  [err] expert %d: count/start mismatch\n", e);
                ok = false; break;
            }
        }
        // check packed rows match embeddings via permutation
        for (int pos = 0; pos < TK && ok; pos++) {
            int t = gpu_perm[pos];
            if (std::memcmp(gpu_packed.data() + (size_t)pos * d,
                            h_emb_u16.data() + (size_t)t * d,
                            d * sizeof(uint16_t)) != 0) {
                std::printf("  [err] packed[%d] != embeddings[%d]\n", pos, t);
                ok = false;
            }
        }
        // check token sets per expert match
        for (int e = 0; e < E && ok; e++) {
            int lo = gpu_estart[e], cnt = gpu_ecount[e];
            std::vector<int> gpu_toks(cnt), cpu_toks(cnt);
            for (int i = 0; i < cnt; i++) {
                gpu_toks[i] = gpu_perm[lo + i];
                cpu_toks[i] = cpu_perm[lo + i];
            }
            std::sort(gpu_toks.begin(), gpu_toks.end());
            std::sort(cpu_toks.begin(), cpu_toks.end());
            if (gpu_toks != cpu_toks) {
                std::printf("  [err] expert %d: token set mismatch\n", e);
                ok = false;
            }
        }

        std::printf("  full pipeline:    %s  (route=%.3fms prefix=%.3fms scatter=%.3fms)\n",
                    ok ? "PASS" : "FAIL", res.route_ms, res.prefix_ms, res.scatter_ms);

        CHECK_CUDA(cudaFree(d_emb));    CHECK_CUDA(cudaFree(d_gate));
        CHECK_CUDA(cudaFree(d_packed)); CHECK_CUDA(cudaFree(d_asgn));
        CHECK_CUDA(cudaFree(d_wt));     CHECK_CUDA(cudaFree(d_estart));
        CHECK_CUDA(cudaFree(d_ecount)); CHECK_CUDA(cudaFree(d_perm));

        if (!ok) return false;
    }

    std::printf("\n  All mini-tests passed.\n");
    return true;
}

// ===========================================================================
// Full run: load .npy embeddings, generate random W_gate, benchmark
// ===========================================================================

static void run_full(const std::string& prefix, int E_cli, const std::string& data_dir) {
    std::printf("\n=== Full run (fused): %s ===\n\n", prefix.c_str());

    NpyArray emb_npy = load_npy(data_dir + "/" + prefix + "_embeddings.npy");
    int T = (int)emb_npy.rows, d = (int)emb_npy.cols;
    int E = E_cli, K = 4;  // match the synthetic data config

    // load assignments just to infer K (don't actually use them — we route on the fly)
    NpyArray asgn_npy = load_npy(data_dir + "/" + prefix + "_assignments.npy");
    K = (int)asgn_npy.cols;

    int TK = T * K;
    std::printf("T=%d  K=%d  E=%d  d=%d  packed_rows=%d\n\n", T, K, E, d, TK);

    // generate a random gate weight matrix [d, E]
    auto h_gate = make_test_gate(d, E, /*seed=*/42);

    const uint16_t* h_emb = (const uint16_t*)emb_npy.data.data();

    // device alloc
    __half*  d_emb;
    float*   d_gate;
    __half*  d_packed;
    int32_t* d_asgn;
    float*   d_wt;
    int32_t* d_estart;
    int32_t* d_ecount;
    int32_t* d_perm;

    CHECK_CUDA(cudaMalloc(&d_emb,    (size_t)T * d * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_gate,   (size_t)d * E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_packed, (size_t)TK * d * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_asgn,   (size_t)TK * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_wt,     (size_t)TK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_estart, E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_ecount, E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_perm,   (size_t)TK * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_emb,  h_emb,          (size_t)T * d * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(),   (size_t)d * E * sizeof(float),   cudaMemcpyHostToDevice));

    // warmup
    for (int i = 0; i < 3; i++)
        fused_pipeline(d_emb, d_gate, T, K, E, d,
                       d_packed, d_estart, d_ecount, d_asgn, d_wt, d_perm);

    // timed runs
    const int NRUNS = 10;
    FusedResult total = {0, 0, 0};
    for (int i = 0; i < NRUNS; i++) {
        auto r = fused_pipeline(d_emb, d_gate, T, K, E, d,
                                d_packed, d_estart, d_ecount, d_asgn, d_wt, d_perm);
        total.route_ms   += r.route_ms;
        total.prefix_ms  += r.prefix_ms;
        total.scatter_ms += r.scatter_ms;
    }

    float avg_route   = total.route_ms / NRUNS;
    float avg_prefix  = total.prefix_ms / NRUNS;
    float avg_scatter = total.scatter_ms / NRUNS;
    float avg_total   = avg_route + avg_prefix + avg_scatter;

    size_t bytes_moved = (size_t)TK * d * sizeof(uint16_t);

    // verify last run: check packed rows match embeddings
    std::vector<uint16_t> gpu_packed(TK * d);
    std::vector<int32_t>  gpu_perm(TK), gpu_ecount(E);
    CHECK_CUDA(cudaMemcpy(gpu_packed.data(), d_packed, (size_t)TK * d * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gpu_perm.data(),   d_perm,   TK * sizeof(int32_t),              cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gpu_ecount.data(), d_ecount, E * sizeof(int32_t),               cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int pos = 0; pos < TK && ok; pos++) {
        int t = gpu_perm[pos];
        if (t < 0 || t >= T) { ok = false; break; }
        if (std::memcmp(gpu_packed.data() + (size_t)pos * d,
                        h_emb + (size_t)t * d,
                        d * sizeof(uint16_t)) != 0) {
            ok = false;
        }
    }

    std::printf("Timing (avg over %d runs):\n", NRUNS);
    std::printf("  route:    %8.3f ms  (gate GEMV + softmax + topK + count)\n", avg_route);
    std::printf("  prefix:   %8.3f ms\n", avg_prefix);
    std::printf("  scatter:  %8.3f ms\n", avg_scatter);
    std::printf("  total:    %8.3f ms  (%.2f GB/s scatter BW)\n",
                avg_total, (bytes_moved / 1e9) / (avg_scatter / 1e3));

    int min_c = INT_MAX, max_c = 0;
    for (int e = 0; e < E; e++) { min_c = std::min(min_c, gpu_ecount[e]); max_c = std::max(max_c, gpu_ecount[e]); }
    std::printf("\nExpert load: min=%d  max=%d  mean=%.1f  imbalance=%.2fx\n",
                min_c, max_c, (double)TK / E, max_c / ((double)TK / E));

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_emb));    CHECK_CUDA(cudaFree(d_gate));
    CHECK_CUDA(cudaFree(d_packed)); CHECK_CUDA(cudaFree(d_asgn));
    CHECK_CUDA(cudaFree(d_wt));     CHECK_CUDA(cudaFree(d_estart));
    CHECK_CUDA(cudaFree(d_ecount)); CHECK_CUDA(cudaFree(d_perm));
}

int main(int argc, char** argv) {
    if (!run_mini_tests()) return 1;

    if (argc > 1) {
        std::string prefix   = argv[1];
        int         E_cli    = (argc > 2) ? std::atoi(argv[2]) : 64;
        std::string data_dir = (argc > 3) ? argv[3] : "results";
        run_full(prefix, E_cli, data_dir);
    }

    return 0;
}
