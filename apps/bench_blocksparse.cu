// apps/bench_blocksparse.cu
//
// Microbench: Megablocks-style "fused gather + W1 GEMM" vs the standard
// pre-packed approach.
//
// In the standard MoE pipeline, scatter materializes a packed buffer
// `bufs.packed[TK, d_model]` whose rows are emb-rows reordered by expert.
// Then a per-expert cuBLAS Hgemm computes mid = packed @ W1[e].
//
// In the Megablocks-style approach, we **never materialize bufs.packed**.
// Instead the FFN GEMM reads input rows directly from `emb` using a
// `sorted_values[]` indirection produced by the scatter. This eliminates
// one full HBM read+write of the packed buffer (TK*d_model*2 * 2 bytes).
//
// This microbench:
//   (A) cuBLAS reference:  mid = packed @ W1[e]   per expert (E calls)
//   (B) Custom kernel:     mid = gather(emb, sorted_values) @ W1[e_for_i]
//                          single launch, gather fused into the matmul
//
// The custom kernel here is intentionally NAIVE (no tensor cores, no
// shared-memory tiling). It demonstrates the technique and verifies
// correctness against cuBLAS. A production version would use CUTLASS's
// GemmUniversalAdapter with a custom GMEM iterator to get tensor-core
// throughput; that's noted as future work in the paper.
//
// Build is wired in the Makefile (bench_blocksparse target).
//
// Run:
//   ./build/bin/bench_blocksparse           # T=2048, uniform
//   ./build/bin/bench_blocksparse zipf      # zipf
//   ./build/bin/bench_blocksparse uniform 16   # decode-shape T=16

#include "moe/fp16_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);}} while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t _bs=(call); if(_bs!=CUBLAS_STATUS_SUCCESS){ \
    std::fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,(int)_bs); std::exit(1);}} while(0)

// ── Shape config (matches Qwen layer-0) ──────────────────────────────────────
struct Shape {
    int T;        // tokens
    int K;        // top-K
    int E;        // experts
    int d_model;  // input dim
    int d_ffn;    // FFN inner dim
};

// ── Synthetic routing (uniform / zipf) ───────────────────────────────────────
// Returns asgn [T*K] with no-duplicate expert per token, plus the resulting
// expert counts (for the packed/sorted layout).
static std::vector<int32_t> make_assignments(
    const Shape& s, const std::string& dist, uint32_t seed)
{
    std::vector<int32_t> asgn((std::size_t)s.T * s.K);
    std::mt19937 rng(seed);
    if (dist == "zipf") {
        std::vector<double> p(s.E);
        double sum = 0;
        for (int e = 0; e < s.E; e++) { p[e] = 1.0 / (e + 1); sum += p[e]; }
        for (int e = 0; e < s.E; e++) p[e] /= sum;
        std::discrete_distribution<int> d(p.begin(), p.end());
        for (int t = 0; t < s.T; t++) {
            std::vector<bool> used(s.E, false);
            for (int k = 0; k < s.K; k++) {
                int e;
                do { e = d(rng); } while (used[e]);
                used[e] = true;
                asgn[(std::size_t)t * s.K + k] = e;
            }
        }
    } else {
        std::uniform_int_distribution<int> u(0, s.E - 1);
        for (int t = 0; t < s.T; t++) {
            std::vector<bool> used(s.E, false);
            for (int k = 0; k < s.K; k++) {
                int e;
                do { e = u(rng); } while (used[e]);
                used[e] = true;
                asgn[(std::size_t)t * s.K + k] = e;
            }
        }
    }
    return asgn;
}

// Build sorted_values + expert_start from asgn. sorted_values[i] = slot_idx
// where slot_idx = t*K + k, sorted by expert. expert_start[e] = first slot
// of expert e in the sorted order.
static void build_sort(const std::vector<int32_t>& asgn, int E,
                       std::vector<int32_t>& sorted_values,
                       std::vector<int32_t>& expert_start,
                       std::vector<int32_t>& expert_count)
{
    int TK = (int)asgn.size();
    expert_count.assign((std::size_t)E, 0);
    for (int i = 0; i < TK; i++) expert_count[asgn[i]]++;

    expert_start.assign((std::size_t)E + 1, 0);
    for (int e = 0; e < E; e++) expert_start[e + 1] = expert_start[e] + expert_count[e];

    std::vector<int32_t> cursor(E, 0);
    sorted_values.assign((std::size_t)TK, 0);
    for (int i = 0; i < TK; i++) {
        int e = asgn[i];
        int pos = expert_start[e] + cursor[e]++;
        sorted_values[pos] = i;  // slot_idx = i = t*K + k
    }
}

// ── (A) Pre-packed cuBLAS reference ──────────────────────────────────────────
// mid[i, :] = packed[i, :] @ W1[e_for_i]
// where packed[i, :] = emb[sorted_values[i] / K, :]
//
// We materialize d_packed first (ordinary fp16 gather), then call cublasHgemm
// per expert. Row-major trick: cublasHgemm(N, N, d_ffn, M_e, d_model, ...).
__global__ void naive_pack_kernel(
    const __half* emb, const int32_t* sorted_values,
    __half* packed,
    int K, int d, int TK)
{
    int i = blockIdx.x;
    int lane = threadIdx.x;
    if (i >= TK) return;
    int t = sorted_values[i] / K;
    const __half* src = emb + (std::size_t)t * d;
    __half* dst = packed + (std::size_t)i * d;
    for (int k = lane; k < d; k += blockDim.x) dst[k] = src[k];
}

static float run_packed_cublas(
    cublasHandle_t h,
    const __half* d_emb, const int32_t* d_sorted_values,
    const __half* d_W1, __half* d_packed, __half* d_mid,
    const std::vector<int32_t>& expert_start,
    const std::vector<int32_t>& expert_count,
    const Shape& s, int warmup, int iters)
{
    int TK = s.T * s.K;
    __half alpha = __float2half(1.f), beta = __float2half(0.f);
    cudaEvent_t a, b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b));

    auto one_pass = [&]() {
        // Pack first.
        naive_pack_kernel<<<TK, 128>>>(d_emb, d_sorted_values, d_packed,
                                       s.K, s.d_model, TK);
        // GEMM per expert.
        for (int e = 0; e < s.E; e++) {
            int M = expert_count[e];
            if (M == 0) continue;
            int off = expert_start[e];
            const __half* A  = d_packed + (std::size_t)off * s.d_model;
            const __half* B1 = d_W1     + (std::size_t)e * s.d_model * s.d_ffn;
            __half*       C  = d_mid    + (std::size_t)off * s.d_ffn;
            CHECK_CUBLAS(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                s.d_ffn, M, s.d_model, &alpha, B1, s.d_ffn, A, s.d_model,
                &beta, C, s.d_ffn));
        }
    };

    for (int i = 0; i < warmup; i++) one_pass();
    CHECK_CUDA(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) one_pass();
    CHECK_CUDA(cudaEventRecord(b));
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    CHECK_CUDA(cudaEventDestroy(a)); CHECK_CUDA(cudaEventDestroy(b));
    return ms / iters;
}

// ── (B) Block-sparse fused gather+W1 kernel ──────────────────────────────────
// One block per (BM-tile of mid rows, BN-tile of mid cols). Each thread
// computes one element of mid. Reads emb rows via sorted_values indirection;
// reads W1[e] for the row's expert (e is determined by binary search on
// expert_start). Naive: no shared memory, no tensor cores.
//
// Layout assumptions:
//   sorted_values [TK]  : maps mid-row i → original slot t*K+k
//   expert_start  [E+1] : expert_start[e..e+1) defines mid-rows owned by e
//
// This kernel is the educational version. Production would use CUTLASS's
// GemmUniversalAdapter with a CustomMmaIteratorA that fetches via indirection.
__global__ void blocksparse_w1_kernel(
    const __half* __restrict__ emb,          // [T, d_model]
    const int32_t* __restrict__ sorted_values,// [TK]
    const __half* __restrict__ W1,           // [E, d_model, d_ffn]
    const int32_t* __restrict__ expert_start,// [E+1]
    __half* __restrict__ mid,                // [TK, d_ffn]
    int E, int K, int d_model, int d_ffn, int TK)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // mid row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // mid col
    if (i >= TK || j >= d_ffn) return;

    // Binary search: which expert owns row i?
    int lo = 0, hi = E;
    while (lo < hi) {
        int mid_e = (lo + hi) >> 1;
        if (expert_start[mid_e + 1] <= i) lo = mid_e + 1;
        else                              hi = mid_e;
    }
    int e = lo;

    int slot = sorted_values[i];
    int t    = slot / K;

    // Dot product: mid[i, j] = sum_k emb[t, k] * W1[e, k, j]
    const __half* emb_row = emb + (std::size_t)t * d_model;
    const __half* W1_e    = W1 + (std::size_t)e * d_model * d_ffn;
    float acc = 0.f;
    for (int k = 0; k < d_model; ++k) {
        float a = __half2float(emb_row[k]);
        float b = __half2float(W1_e[(std::size_t)k * d_ffn + j]);
        acc += a * b;
    }
    mid[(std::size_t)i * d_ffn + j] = __float2half(acc);
}

static float run_blocksparse_naive(
    const __half* d_emb, const int32_t* d_sorted_values,
    const __half* d_W1, const int32_t* d_expert_start,
    __half* d_mid, const Shape& s, int warmup, int iters)
{
    int TK = s.T * s.K;
    cudaEvent_t a, b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b));

    // 1 thread per output element. Block (32 cols, 4 rows) = 128 threads.
    dim3 block(32, 4);
    dim3 grid((s.d_ffn + 31) / 32, (TK + 3) / 4);

    auto one_pass = [&]() {
        blocksparse_w1_kernel<<<grid, block>>>(
            d_emb, d_sorted_values, d_W1, d_expert_start, d_mid,
            s.E, s.K, s.d_model, s.d_ffn, TK);
    };

    for (int i = 0; i < warmup; i++) one_pass();
    CHECK_CUDA(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) one_pass();
    CHECK_CUDA(cudaEventRecord(b));
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    CHECK_CUDA(cudaEventDestroy(a)); CHECK_CUDA(cudaEventDestroy(b));
    return ms / iters;
}

// ── Correctness: max abs diff between reference and custom on a sample ──────
static double max_abs_diff(const std::vector<uint16_t>& a,
                           const std::vector<uint16_t>& b, std::size_t n)
{
    double maxd = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = std::abs(moe::h2f(a[i]) - moe::h2f(b[i]));
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string dist = (argc > 1) ? argv[1] : "uniform";
    int T_arg = (argc > 2) ? std::atoi(argv[2]) : 2048;

    Shape s{T_arg, 4, 64, 2048, 8192};
    int TK = s.T * s.K;

    std::printf("\n== bench_blocksparse  dist=%s  T=%d  K=%d  E=%d  d=%d  d_ffn=%d ==\n\n",
                dist.c_str(), s.T, s.K, s.E, s.d_model, s.d_ffn);

    // Host-side: routing + sorted layout + W1 weights
    auto h_asgn = make_assignments(s, dist, 42);
    std::vector<int32_t> h_sorted_values, h_expert_start, h_expert_count;
    build_sort(h_asgn, s.E, h_sorted_values, h_expert_start, h_expert_count);

    int min_c = INT32_MAX, max_c = 0;
    for (int e = 0; e < s.E; ++e) {
        min_c = std::min(min_c, h_expert_count[e]);
        max_c = std::max(max_c, h_expert_count[e]);
    }
    std::printf("   expert load: min=%d max=%d mean=%.1f imbalance=%.2fx\n\n",
                min_c, max_c, (double)TK / s.E, (double)max_c * s.E / TK);

    // Host-side weights (small range so fp16 doesn't overflow on dot products)
    std::vector<uint16_t> h_emb((std::size_t)s.T * s.d_model);
    std::vector<uint16_t> h_W1((std::size_t)s.E * s.d_model * s.d_ffn);
    moe::fill_random_half(h_emb.data(), h_emb.size(), 1, /*half_range=*/0.05f);
    moe::fill_random_half(h_W1.data(), h_W1.size(), 2, /*half_range=*/0.005f);

    // Device allocations
    __half*  d_emb = nullptr;     CHECK_CUDA(cudaMalloc(&d_emb, h_emb.size() * 2));
    __half*  d_W1 = nullptr;      CHECK_CUDA(cudaMalloc(&d_W1, h_W1.size() * 2));
    int32_t* d_sorted_values = nullptr;
    int32_t* d_expert_start = nullptr;
    __half*  d_packed = nullptr;
    __half*  d_mid_ref = nullptr;
    __half*  d_mid_bs  = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sorted_values, TK * 4));
    CHECK_CUDA(cudaMalloc(&d_expert_start,  (s.E + 1) * 4));
    CHECK_CUDA(cudaMalloc(&d_packed, (std::size_t)TK * s.d_model * 2));
    CHECK_CUDA(cudaMalloc(&d_mid_ref, (std::size_t)TK * s.d_ffn * 2));
    CHECK_CUDA(cudaMalloc(&d_mid_bs,  (std::size_t)TK * s.d_ffn * 2));

    CHECK_CUDA(cudaMemcpy(d_emb, h_emb.data(), h_emb.size() * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1,  h_W1.data(),  h_W1.size() * 2,  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sorted_values, h_sorted_values.data(), TK * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_start,  h_expert_start.data(), (s.E + 1) * 4, cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    // Warmup + time both variants
    int warmup = 3, iters = 10;

    // (A) packed cuBLAS
    CHECK_CUDA(cudaMemset(d_mid_ref, 0, (std::size_t)TK * s.d_ffn * 2));
    float ms_packed = run_packed_cublas(
        cublas, d_emb, d_sorted_values, d_W1, d_packed, d_mid_ref,
        h_expert_start, h_expert_count, s, warmup, iters);

    // (B) block-sparse naive
    CHECK_CUDA(cudaMemset(d_mid_bs, 0, (std::size_t)TK * s.d_ffn * 2));
    float ms_bs = run_blocksparse_naive(
        d_emb, d_sorted_values, d_W1, d_expert_start, d_mid_bs, s, warmup, iters);

    // Correctness: compare a sample of mid_ref vs mid_bs
    std::size_t sample = std::min<std::size_t>((std::size_t)TK * s.d_ffn,
                                               (std::size_t)64 * s.d_ffn);
    std::vector<uint16_t> h_ref(sample), h_bs(sample);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_mid_ref, sample * 2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_bs.data(),  d_mid_bs,  sample * 2, cudaMemcpyDeviceToHost));
    double diff = max_abs_diff(h_ref, h_bs, sample);
    std::printf("   correctness   max|ref - blocksparse| = %.4e  (sample=%zu rows)\n",
                diff, sample / (std::size_t)s.d_ffn);

    // FLOPS comparison (should match — same algebraic op)
    double flops = 2.0 * (double)TK * s.d_ffn * s.d_model;

    std::printf("\n   variant            time (ms)   throughput (TFLOPS)   ratio vs packed\n");
    std::printf("   packed cuBLAS      %8.3f        %7.2f               1.000x\n",
                ms_packed, flops / 1e12 / (ms_packed / 1e3));
    std::printf("   blocksparse naive  %8.3f        %7.2f               %.3fx\n\n",
                ms_bs, flops / 1e12 / (ms_bs / 1e3), ms_packed / ms_bs);

    std::printf("   memory: packed buffer materialization saved by blocksparse:\n"
                "      %.2f MB write + %.2f MB read = %.2f MB / iteration\n\n",
                (double)TK * s.d_model * 2 / (1024.0 * 1024.0),
                (double)TK * s.d_model * 2 / (1024.0 * 1024.0),
                2.0 * TK * s.d_model * 2 / (1024.0 * 1024.0));

    cublasDestroy(cublas);
    cudaFree(d_emb); cudaFree(d_W1);
    cudaFree(d_sorted_values); cudaFree(d_expert_start);
    cudaFree(d_packed); cudaFree(d_mid_ref); cudaFree(d_mid_bs);
    return 0;
}
