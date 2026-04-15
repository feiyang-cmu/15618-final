// apps/bench_e2e.cu
//
// End-to-end single-GPU MoE layer benchmark.
//
// Pipeline: route → scatter (via IScatter) → expert GEMMs (cuBLAS hgemm) → unpack+combine
//
// Two routing modes:
//   naive: gate_kernel → softmax_kernel → topk_kernel  (3 launches, logits/probs through HBM)
//   fused: fused_route_kernel                          (1 launch, logits in smem)
//
// Scatter goes through moe::IScatter so swapping atomic/sort/warp strategies is
// a single CLI flag.

#include "moe/cpu_reference.hpp"
#include "moe/fp16_host.hpp"
#include "moe/npy_io.hpp"
#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvToolsExt.h>

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _st = (call);                                           \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n",                   \
                         __FILE__, __LINE__, (int)_st);                        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ============================================================================
// Routing kernels (produce assignments[T,K] and routing_weights[T,K])
// ============================================================================

namespace {

__global__ void gate_kernel(const __half* __restrict__ emb,
                            const float*  __restrict__ W_gate,
                            float*        __restrict__ logits,
                            int T, int E, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * E) return;
    int t = idx / E, e = idx % E;
    float acc = 0.f;
    for (int i = 0; i < d; ++i) acc += __half2float(emb[t * d + i]) * W_gate[i * E + e];
    logits[t * E + e] = acc;
}

__global__ void softmax_kernel(const float* __restrict__ logits,
                               float*       __restrict__ probs,
                               int T, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* row = logits + t * E;
    float*       out = probs  + t * E;
    float mx = row[0];
    for (int e = 1; e < E; ++e) mx = fmaxf(mx, row[e]);
    float sum = 0.f;
    for (int e = 0; e < E; ++e) { out[e] = expf(row[e] - mx); sum += out[e]; }
    for (int e = 0; e < E; ++e) out[e] /= sum;
}

__global__ void topk_kernel(const float* __restrict__ probs,
                            int32_t*     __restrict__ assignments,
                            float*       __restrict__ weights,
                            int T, int K, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float lp[128];  // assumes E <= 128
    for (int e = 0; e < E; ++e) lp[e] = probs[t * E + e];
    for (int k = 0; k < K; ++k) {
        int best = 0;
        for (int e = 1; e < E; ++e) if (lp[e] > lp[best]) best = e;
        assignments[t * K + k] = best;
        weights[t * K + k]     = lp[best];
        lp[best] = -1.f;
    }
}

__global__ void fused_route_kernel(const __half* __restrict__ emb,
                                   const float*  __restrict__ W_gate,
                                   int T, int K, int E, int d,
                                   int32_t* __restrict__ assignments,
                                   float*   __restrict__ weights) {
    int t = blockIdx.x, e = threadIdx.x;
    if (t >= T || e >= E) return;

    float logit = 0.f;
    for (int i = 0; i < d; ++i) logit += __half2float(emb[t * d + i]) * W_gate[i * E + e];

    extern __shared__ float smem[];
    smem[e] = logit;
    __syncthreads();

    __shared__ int32_t s_idx[8];
    __shared__ float   s_val[8];
    if (e == 0) {
        float mx = smem[0];
        for (int i = 1; i < E; ++i) mx = fmaxf(mx, smem[i]);
        float sum = 0.f;
        for (int i = 0; i < E; ++i) { smem[i] = expf(smem[i] - mx); sum += smem[i]; }
        for (int i = 0; i < E; ++i) smem[i] /= sum;
        for (int k = 0; k < K; ++k) {
            int best = 0;
            for (int i = 1; i < E; ++i) if (smem[i] > smem[best]) best = i;
            s_idx[k] = best; s_val[k] = smem[best]; smem[best] = -1.f;
        }
    }
    __syncthreads();

    if (e < K) {
        assignments[t * K + e] = s_idx[e];
        weights[t * K + e]     = s_val[e];
    }
}

// ============================================================================
// Unpack + combine: expert_out[TK, d] → output[T, d] using perm & packed_weights
// ============================================================================

__global__ void unpack_combine_kernel(const __half*  __restrict__ expert_out,
                                      const float*   __restrict__ packed_weights,
                                      const int32_t* __restrict__ perm,
                                      float*         __restrict__ output,  // [T, d] zeroed
                                      int TK, int d) {
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

} // namespace

// ============================================================================
// E2E run — single forward pass
// ============================================================================

struct E2EResult {
    float route_ms;
    float pack_ms;
    float gemm_ms;   // gemm + unpack (events straddle the D2H sync inside pack)
    float total_ms;
};

struct E2EBuffers {
    // routing
    float*        d_logits = nullptr;   // only used in naive
    float*        d_probs  = nullptr;   // only used in naive
    int32_t*      d_asgn   = nullptr;
    float*        d_rwt    = nullptr;
    // scatter I/O
    moe::MoEBuffers scatter_bufs{};
    // expert GEMM workspace
    __half*       d_expert_out = nullptr;
    __half*       d_ffn_tmp    = nullptr;
    float*        d_out_f32    = nullptr;
    __half*       d_output     = nullptr;

    void free() {
        cudaFree(d_logits); cudaFree(d_probs);
        cudaFree(d_asgn);   cudaFree(d_rwt);
        cudaFree(scatter_bufs.packed);        cudaFree(scatter_bufs.packed_weights);
        cudaFree(scatter_bufs.perm);          cudaFree(scatter_bufs.expert_count);
        cudaFree(scatter_bufs.expert_start);  cudaFree(scatter_bufs.strategy_ws);
        cudaFree(d_expert_out); cudaFree(d_ffn_tmp);
        cudaFree(d_out_f32);    cudaFree(d_output);
    }
};

static E2EBuffers allocate_e2e(const moe::MoEParams& p, std::size_t scatter_ws, bool naive) {
    E2EBuffers b;
    const int TK = p.T * p.K;
    if (naive) {
        b.d_logits = moe::device_alloc<float>((std::size_t)p.T * p.E);
        b.d_probs  = moe::device_alloc<float>((std::size_t)p.T * p.E);
    }
    b.d_asgn = moe::device_alloc<int32_t>((std::size_t)TK);
    b.d_rwt  = moe::device_alloc<float>((std::size_t)TK);

    b.scatter_bufs.packed         = moe::device_alloc<__half>((std::size_t)TK * p.d_model);
    b.scatter_bufs.packed_weights = moe::device_alloc<float>((std::size_t)TK);
    b.scatter_bufs.perm           = moe::device_alloc<int32_t>((std::size_t)TK);
    b.scatter_bufs.expert_count   = moe::device_alloc<int32_t>((std::size_t)p.E);
    b.scatter_bufs.expert_start   = moe::device_alloc<int32_t>((std::size_t)(p.E + 1));
    b.scatter_bufs.strategy_ws_bytes = scatter_ws;
    MOE_CUDA_CHECK(cudaMalloc(&b.scatter_bufs.strategy_ws, scatter_ws));

    b.d_expert_out = moe::device_alloc<__half>((std::size_t)TK * p.d_model);
    b.d_ffn_tmp    = moe::device_alloc<__half>((std::size_t)TK * p.d_ffn);
    b.d_out_f32    = moe::device_alloc<float>((std::size_t)p.T * p.d_model);
    b.d_output     = moe::device_alloc<__half>((std::size_t)p.T * p.d_model);
    return b;
}

// Full forward pass. Caller owns all device memory (passed in `bufs`).
static E2EResult run_e2e(const __half* d_emb,
                         const float*  d_W_gate,
                         const __half* d_W1,
                         const __half* d_W2,
                         const moe::MoEParams& p,
                         bool use_fused,
                         moe::IScatter& scatter,
                         E2EBuffers& bufs,
                         cublasHandle_t cublas,
                         cudaStream_t stream) {
    const int TK  = p.T * p.K;
    const int blk = 256;

    cudaEvent_t ev[5];
    for (int i = 0; i < 5; ++i) MOE_CUDA_CHECK(cudaEventCreate(&ev[i]));

    // ---- ROUTE ----
    MOE_CUDA_CHECK(cudaEventRecord(ev[0], stream));
    if (use_fused) {
        nvtxRangePush("fused_route");
        fused_route_kernel<<<p.T, p.E, p.E * sizeof(float), stream>>>(
            d_emb, d_W_gate, p.T, p.K, p.E, p.d_model, bufs.d_asgn, bufs.d_rwt);
        nvtxRangePop();
    } else {
        nvtxRangePush("gate");
        gate_kernel<<<((p.T * p.E) + blk - 1) / blk, blk, 0, stream>>>(
            d_emb, d_W_gate, bufs.d_logits, p.T, p.E, p.d_model);
        nvtxRangePop();
        nvtxRangePush("softmax");
        softmax_kernel<<<(p.T + blk - 1) / blk, blk, 0, stream>>>(
            bufs.d_logits, bufs.d_probs, p.T, p.E);
        nvtxRangePop();
        nvtxRangePush("topk");
        topk_kernel<<<(p.T + blk - 1) / blk, blk, 0, stream>>>(
            bufs.d_probs, bufs.d_asgn, bufs.d_rwt, p.T, p.K, p.E);
        nvtxRangePop();
    }
    MOE_CUDA_CHECK(cudaEventRecord(ev[1], stream));

    // ---- SCATTER (via IScatter) ----
    nvtxRangePush("scatter");
    scatter.run(d_emb, bufs.d_asgn, bufs.d_rwt, bufs.scatter_bufs, p, stream);
    nvtxRangePop();
    MOE_CUDA_CHECK(cudaEventRecord(ev[2], stream));

    // D2H copy of per-expert counts for the GEMM loop.
    std::vector<int32_t> h_estart(p.E), h_ecount(p.E);
    MOE_CUDA_CHECK(cudaStreamSynchronize(stream));
    MOE_CUDA_CHECK(cudaMemcpy(h_estart.data(), bufs.scatter_bufs.expert_start,
                              p.E * sizeof(int32_t), cudaMemcpyDeviceToHost));
    MOE_CUDA_CHECK(cudaMemcpy(h_ecount.data(), bufs.scatter_bufs.expert_count,
                              p.E * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // ---- EXPERT GEMMs (sequential cuBLAS hgemm per expert) ----
    // Row-major trick: cublas computes col-major, so pass A[M,K] as col-major
    // [K,M]. For C[M,N] = A[M,K] @ B[K,N] use cublas(N, M, K, B, N, A, K, C, N).
    __half alpha_h = __float2half(1.f), beta_h = __float2half(0.f);
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    MOE_CUDA_CHECK(cudaEventRecord(ev[3], stream));
    nvtxRangePush("expert_gemms");
    for (int e = 0; e < p.E; ++e) {
        int cnt = h_ecount[e];
        if (cnt == 0) continue;
        int off = h_estart[e];

        const __half* A  = bufs.scatter_bufs.packed + (size_t)off * p.d_model;
        const __half* B1 = d_W1 + (size_t)e * p.d_model * p.d_ffn;
        const __half* B2 = d_W2 + (size_t)e * p.d_ffn   * p.d_model;
        __half* mid      = bufs.d_ffn_tmp    + (size_t)off * p.d_ffn;
        __half* out      = bufs.d_expert_out + (size_t)off * p.d_model;

        // up-proj [cnt, d_model] @ [d_model, d_ffn] -> [cnt, d_ffn]
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            p.d_ffn, cnt, p.d_model, &alpha_h,
            B1, p.d_ffn, A, p.d_model,
            &beta_h, mid, p.d_ffn));

        // down-proj [cnt, d_ffn] @ [d_ffn, d_model] -> [cnt, d_model]
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            p.d_model, cnt, p.d_ffn, &alpha_h,
            B2, p.d_model, mid, p.d_ffn,
            &beta_h, out, p.d_model));
    }
    nvtxRangePop();

    // ---- UNPACK + COMBINE ----
    moe::device_zero_async(bufs.d_out_f32, (std::size_t)p.T * p.d_model, stream);
    nvtxRangePush("unpack_combine");
    unpack_combine_kernel<<<TK, blk, 0, stream>>>(
        bufs.d_expert_out, bufs.scatter_bufs.packed_weights,
        bufs.scatter_bufs.perm, bufs.d_out_f32, TK, p.d_model);
    nvtxRangePop();
    nvtxRangePush("f32_to_f16");
    f32_to_f16_kernel<<<(p.T * p.d_model + blk - 1) / blk, blk, 0, stream>>>(
        bufs.d_out_f32, bufs.d_output, p.T * p.d_model);
    nvtxRangePop();

    MOE_CUDA_CHECK(cudaEventRecord(ev[4], stream));
    MOE_CUDA_CHECK(cudaStreamSynchronize(stream));

    E2EResult r;
    MOE_CUDA_CHECK(cudaEventElapsedTime(&r.route_ms, ev[0], ev[1]));
    MOE_CUDA_CHECK(cudaEventElapsedTime(&r.pack_ms,  ev[1], ev[2]));
    MOE_CUDA_CHECK(cudaEventElapsedTime(&r.gemm_ms,  ev[3], ev[4]));
    r.total_ms = r.route_ms + r.pack_ms + r.gemm_ms;
    for (int i = 0; i < 5; ++i) cudaEventDestroy(ev[i]);
    return r;
}

// ============================================================================
// Mini-test + full benchmark
// ============================================================================

static bool run_mini_test(const std::string& strategy_name) {
    std::printf("\n=== Mini-test (T=16 K=2 E=4 d=8 d_ffn=32 strategy=%s) ===\n\n",
                strategy_name.c_str());
    moe::MoEParams p{/*T=*/16, /*K=*/2, /*E=*/4, /*d_model=*/8, /*d_ffn=*/32};

    std::vector<uint16_t> h_emb((std::size_t)p.T * p.d_model);
    std::vector<float>    h_gate((std::size_t)p.d_model * p.E);
    std::vector<uint16_t> h_W1((std::size_t)p.E * p.d_model * p.d_ffn);
    std::vector<uint16_t> h_W2((std::size_t)p.E * p.d_ffn   * p.d_model);
    moe::fill_random_half (h_emb.data(),  h_emb.size(),  1);
    moe::fill_random_float(h_gate.data(), h_gate.size(), 2);
    moe::fill_random_half (h_W1.data(),   h_W1.size(),   3);
    moe::fill_random_half (h_W2.data(),   h_W2.size(),   4);

    __half *d_emb, *d_W1, *d_W2; float* d_gate;
    MOE_CUDA_CHECK(cudaMalloc(&d_emb,  h_emb.size()  * 2));
    MOE_CUDA_CHECK(cudaMalloc(&d_gate, h_gate.size() * 4));
    MOE_CUDA_CHECK(cudaMalloc(&d_W1,   h_W1.size()   * 2));
    MOE_CUDA_CHECK(cudaMalloc(&d_W2,   h_W2.size()   * 2));
    MOE_CUDA_CHECK(cudaMemcpy(d_emb,  h_emb.data(),  h_emb.size()  * 2, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), h_gate.size() * 4, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_W1,   h_W1.data(),   h_W1.size()   * 2, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_W2,   h_W2.data(),   h_W2.size()   * 2, cudaMemcpyHostToDevice));

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    auto scatter = moe::make_scatter(strategy_name);
    if (!scatter) { std::fprintf(stderr, "unknown strategy '%s'\n", strategy_name.c_str()); return false; }

    for (int mode = 0; mode < 2; ++mode) {
        bool fused = (mode == 1);
        E2EBuffers b = allocate_e2e(p, scatter->workspace_bytes(p), /*naive=*/!fused);
        auto r = run_e2e(d_emb, d_gate, d_W1, d_W2, p, fused, *scatter, b, cublas, 0);
        std::printf("  %-6s route=%.3f  pack=%.3f  gemm+unpack=%.3f  total=%.3f ms\n",
                    fused ? "fused" : "naive", r.route_ms, r.pack_ms, r.gemm_ms, r.total_ms);
        b.free();
    }

    cublasDestroy(cublas);
    cudaFree(d_emb); cudaFree(d_gate); cudaFree(d_W1); cudaFree(d_W2);
    return true;
}

static void run_full(const std::string& prefix, int E_cli,
                     const std::string& data_dir,
                     const std::string& strategy_name) {
    std::printf("\n=== E2E MoE Layer Benchmark (strategy=%s) ===\n\n", strategy_name.c_str());

    auto emb  = moe::load_npy(data_dir + "/" + prefix + "_embeddings.npy");
    auto asgn = moe::load_npy(data_dir + "/" + prefix + "_assignments.npy");
    if (emb.dtype != "<f2" || asgn.dtype != "<i4") {
        std::fprintf(stderr, "unexpected dtypes\n"); return;
    }

    moe::MoEParams p{};
    p.T       = (int)emb.rows;
    p.K       = (int)asgn.cols;
    p.E       = E_cli;
    p.d_model = (int)emb.cols;
    p.d_ffn   = p.d_model * 4;
    const int TK = p.T * p.K;

    std::printf("T=%d K=%d E=%d d=%d d_ffn=%d packed_rows=%d\n",
                p.T, p.K, p.E, p.d_model, p.d_ffn, TK);

    // Memory sanity check.
    std::size_t w1 = (std::size_t)p.E * p.d_model * p.d_ffn * 2;
    std::size_t w2 = (std::size_t)p.E * p.d_ffn   * p.d_model * 2;
    std::size_t free_mem = 0, total_mem = 0;
    MOE_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::printf("Expert weights: W1=%.1f MB  W2=%.1f MB  (GPU free=%.1f MB)\n\n",
                w1 / 1e6, w2 / 1e6, free_mem / 1e6);

    // Generate random expert weights + gate.
    std::vector<float>    h_gate((std::size_t)p.d_model * p.E);
    std::vector<uint16_t> h_W1((std::size_t)p.E * p.d_model * p.d_ffn);
    std::vector<uint16_t> h_W2((std::size_t)p.E * p.d_ffn   * p.d_model);
    moe::fill_random_float(h_gate.data(), h_gate.size(), 42);
    moe::fill_random_half (h_W1.data(),   h_W1.size(),   100);
    moe::fill_random_half (h_W2.data(),   h_W2.size(),   200);

    __half *d_emb, *d_W1, *d_W2; float* d_gate;
    MOE_CUDA_CHECK(cudaMalloc(&d_emb,  (std::size_t)p.T * p.d_model * 2));
    MOE_CUDA_CHECK(cudaMalloc(&d_gate, (std::size_t)p.d_model * p.E * 4));
    MOE_CUDA_CHECK(cudaMalloc(&d_W1,   w1));
    MOE_CUDA_CHECK(cudaMalloc(&d_W2,   w2));
    MOE_CUDA_CHECK(cudaMemcpy(d_emb,  emb.data.data(),  (std::size_t)p.T * p.d_model * 2, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(),    h_gate.size() * 4, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_W1,   h_W1.data(),      w1, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_W2,   h_W2.data(),      w2, cudaMemcpyHostToDevice));
    h_W1.clear(); h_W1.shrink_to_fit();
    h_W2.clear(); h_W2.shrink_to_fit();

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    auto scatter = moe::make_scatter(strategy_name);
    if (!scatter) { std::fprintf(stderr, "unknown strategy '%s'\n", strategy_name.c_str()); return; }

    // Separate buffer set for each routing mode so we can keep them hot across
    // the warmup + timed loop.
    E2EBuffers b_naive = allocate_e2e(p, scatter->workspace_bytes(p), /*naive=*/true);
    E2EBuffers b_fused = allocate_e2e(p, scatter->workspace_bytes(p), /*naive=*/false);

    // Warmup.
    for (int m = 0; m < 2; ++m) {
        run_e2e(d_emb, d_gate, d_W1, d_W2, p, /*fused=*/m == 1,
                *scatter, m == 1 ? b_fused : b_naive, cublas, 0);
    }

    // Timed: N runs per mode, report per-stage averages + totals.
    constexpr int NRUNS = 10;
    for (int m = 0; m < 2; ++m) {
        bool fused = (m == 1);
        E2EResult tot{};
        for (int i = 0; i < NRUNS; ++i) {
            auto r = run_e2e(d_emb, d_gate, d_W1, d_W2, p, fused,
                             *scatter, fused ? b_fused : b_naive, cublas, 0);
            tot.route_ms += r.route_ms; tot.pack_ms += r.pack_ms;
            tot.gemm_ms  += r.gemm_ms;  tot.total_ms += r.total_ms;
        }
        float route = tot.route_ms / NRUNS, pack = tot.pack_ms / NRUNS;
        float gemm  = tot.gemm_ms / NRUNS,  total = tot.total_ms / NRUNS;
        float pack_frac = (route + pack) / total * 100.f;

        std::printf("[%s] avg over %d runs\n", fused ? "FUSED" : "NAIVE", NRUNS);
        std::printf("  route:        %8.3f ms  (%5.1f%%)\n", route, route / total * 100);
        std::printf("  pack:         %8.3f ms  (%5.1f%%)\n", pack,  pack  / total * 100);
        std::printf("  gemm+unpack:  %8.3f ms  (%5.1f%%)\n", gemm,  gemm  / total * 100);
        std::printf("  total:        %8.3f ms\n", total);
        std::printf("  packing frac: %5.1f%%  (route+pack)\n\n", pack_frac);
    }

    b_naive.free(); b_fused.free();
    cublasDestroy(cublas);
    cudaFree(d_emb); cudaFree(d_gate); cudaFree(d_W1); cudaFree(d_W2);
}

int main(int argc, char** argv) {
    std::string prefix    = (argc > 1) ? argv[1] : "";
    int         E_cli     = (argc > 2) ? std::atoi(argv[2]) : 64;
    std::string data_dir  = (argc > 3) ? argv[3] : "results";
    std::string strategy  = (argc > 4) ? argv[4] : "atomic";

    if (!run_mini_test(strategy)) return 1;
    if (!prefix.empty()) run_full(prefix, E_cli, data_dir, strategy);
    return 0;
}
