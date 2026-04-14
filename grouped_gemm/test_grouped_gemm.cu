// grouped_gemm/test_grouped_gemm.cu
//
// Standalone sanity check: CUTLASS GemmGrouped vs sequential cuBLAS hgemm.
// Mimics the per-expert up-projection of an MoE layer.
//
// For each of E experts with variable token counts M_e:
//   mid[M_e, d_ffn] = packed[M_e, d] @ W1[d, d_ffn]   (row-major)
//
// We run it two ways:
//   (1) sequential cuBLAS: E calls of cublasHgemm, one per expert
//   (2) CUTLASS grouped GEMM: a single kernel launch for all E problems
// Verify outputs match, then time each approach.
//
// Build:
//   CUBLAS12=$HOME/miniconda3/envs/myenv/lib/python3.9/site-packages/nvidia/cublas
//   CUTLASS=third_party/cutlass
//   nvcc -O3 -std=c++17 -arch=sm_80 \
//     -I${CUTLASS}/include -I${CUTLASS}/tools/util/include \
//     -I${CUBLAS12}/include \
//     grouped_gemm/test_grouped_gemm.cu -o grouped_gemm/test_grouped_gemm \
//     -Xlinker -rpath=${CUBLAS12}/lib \
//     -Xlinker ${CUBLAS12}/lib/libcublas.so.12 \
//     -Xlinker ${CUBLAS12}/lib/libcublasLt.so.12
//
// Run:
//   ./grouped_gemm/test_grouped_gemm          # uniform distribution
//   ./grouped_gemm/test_grouped_gemm zipf     # zipf distribution

#include <algorithm>
#include <cassert>
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

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);}} while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS){ \
    std::fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,(int)s); std::exit(1);}} while(0)

// ------- fp16 <-> float host helpers -------
static uint16_t f2h(float val) {
    uint32_t f; std::memcpy(&f,&val,4);
    uint32_t sign=(f>>31)&1; int32_t exp=((f>>23)&0xff)-127; uint32_t mant=f&0x7fffff;
    if(exp>15) return (uint16_t)((sign<<15)|0x7c00);
    if(exp<-14) return (uint16_t)(sign<<15);
    return (uint16_t)((sign<<15)|((exp+15)<<10)|(mant>>13));
}
static float h2f(uint16_t h) {
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1f, m=h&0x3ff, f;
    if(e==0){ if(m==0) f=s<<31; else{e=1;while(!(m&0x400)){m<<=1;e--;}m&=0x3ff;f=(s<<31)|((e+127-15)<<23)|(m<<13);}}
    else if(e==31) f=(s<<31)|0x7f800000|(m<<13);
    else f=(s<<31)|((e+127-15)<<23)|(m<<13);
    float r; std::memcpy(&r,&f,4); return r;
}

// ---------------------------------------------------------------------------
// CUTLASS grouped GEMM types (RowMajor all around, tensor op, sm_80)
// ---------------------------------------------------------------------------
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAcc = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Tile shape tunable via compile-time macro: TILE_M, TILE_N, TILE_K.
// 128x128x32 is the CUTLASS grouped default; 64x64x32 wastes less for small M.
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 32
#endif
#ifndef WARP_M
#define WARP_M 64
#endif
#ifndef WARP_N
#define WARP_N 64
#endif
#ifndef STAGES
#define STAGES 4
#endif

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementC, LayoutC,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TILE_M, TILE_N, TILE_K>,
    cutlass::gemm::GemmShape<WARP_M, WARP_N, TILE_K>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 8, ElementAcc, ElementAcc>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    STAGES>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// ---------------------------------------------------------------------------
// Build per-expert token counts.  Uniform -> all near mean; zipf -> heavy tail.
// ---------------------------------------------------------------------------
static std::vector<int> make_counts(int E, int TK, const std::string& dist, uint32_t seed) {
    std::vector<int> m(E, 0);
    std::mt19937 rng(seed);
    if (dist == "zipf") {
        // zipf(1.0) over E experts, then multinomial draw of TK tokens
        std::vector<double> p(E);
        double s = 0;
        for (int e = 0; e < E; e++) { p[e] = 1.0 / (e + 1); s += p[e]; }
        for (int e = 0; e < E; e++) p[e] /= s;
        std::discrete_distribution<int> d(p.begin(), p.end());
        for (int i = 0; i < TK; i++) m[d(rng)]++;
    } else {
        // uniform: multinomial with equal probs
        std::uniform_int_distribution<int> u(0, E - 1);
        for (int i = 0; i < TK; i++) m[u(rng)]++;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Ref: sequential cuBLAS hgemm, one call per expert.
// Row-major trick: cuBLAS is col-major, so pass (B^T, A^T) to compute C^T.
//   cublasHgemm(OP_N, OP_N, N, M, K, B, N, A, K, C, N)
// ---------------------------------------------------------------------------
static float run_sequential_cublas(
    cublasHandle_t h,
    const __half* d_packed, const __half* d_W1, __half* d_mid,
    const std::vector<int>& M, const std::vector<int>& offA,
    int d, int d_ffn, int warmups, int iters)
{
    int E = (int)M.size();
    __half alpha = __float2half(1.f), beta = __float2half(0.f);

    cudaEvent_t a, b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b));

    auto one_pass = [&]() {
        for (int e = 0; e < E; e++) {
            if (M[e] == 0) continue;
            const __half* A  = d_packed + (size_t)offA[e] * d;
            const __half* B1 = d_W1     + (size_t)e * d * d_ffn;
            __half*       C  = d_mid    + (size_t)offA[e] * d_ffn;
            // row-major: C[M,N] = A[M,K] @ B[K,N]   (K=d, N=d_ffn, M=M[e])
            CHECK_CUBLAS(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                d_ffn, M[e], d, &alpha, B1, d_ffn, A, d, &beta, C, d_ffn));
        }
    };

    for (int i = 0; i < warmups; i++) one_pass();
    CHECK_CUDA(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) one_pass();
    CHECK_CUDA(cudaEventRecord(b));
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    CHECK_CUDA(cudaEventDestroy(a)); CHECK_CUDA(cudaEventDestroy(b));
    return ms / iters;
}

// ---------------------------------------------------------------------------
// CUTLASS grouped GEMM: all E problems in one kernel launch.
// Per-problem (M, N, K) varies (M_e varies, K=d, N=d_ffn fixed).
// ---------------------------------------------------------------------------
static float run_grouped_cutlass(
    const __half* d_packed, const __half* d_W1, __half* d_mid,
    const std::vector<int>& M, const std::vector<int>& offA,
    int d, int d_ffn, int warmups, int iters)
{
    int E = (int)M.size();

    // Build per-problem metadata on the host, then copy to device.
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementA*> h_ptrA;
    std::vector<ElementB*> h_ptrB;
    std::vector<ElementC*> h_ptrC;
    std::vector<int64_t> h_lda, h_ldb, h_ldc;
    problem_sizes.reserve(E); h_ptrA.reserve(E); h_ptrB.reserve(E); h_ptrC.reserve(E);
    h_lda.reserve(E); h_ldb.reserve(E); h_ldc.reserve(E);

    for (int e = 0; e < E; e++) {
        int m = M[e];
        if (m == 0) continue;  // skip empty experts; grouped kernel doesn't like M=0
        problem_sizes.emplace_back(m, d_ffn, d);
        h_ptrA.push_back( (ElementA*)(d_packed + (size_t)offA[e] * d) );
        h_ptrB.push_back( (ElementB*)(d_W1     + (size_t)e * d * d_ffn) );
        h_ptrC.push_back( (ElementC*)(d_mid    + (size_t)offA[e] * d_ffn) );
        h_lda.push_back(d);       // A[m, d] row-major -> lda = d
        h_ldb.push_back(d_ffn);   // B[d, d_ffn] row-major -> ldb = d_ffn
        h_ldc.push_back(d_ffn);   // C[m, d_ffn] row-major -> ldc = d_ffn
    }
    int n_problems = (int)problem_sizes.size();

    // Upload metadata arrays.
    cutlass::gemm::GemmCoord* d_problem_sizes = nullptr;
    ElementA** d_ptrA = nullptr; ElementB** d_ptrB = nullptr; ElementC** d_ptrC = nullptr;
    int64_t* d_lda = nullptr; int64_t* d_ldb = nullptr; int64_t* d_ldc = nullptr;
    size_t ps_bytes = sizeof(cutlass::gemm::GemmCoord) * n_problems;
    CHECK_CUDA(cudaMalloc(&d_problem_sizes, ps_bytes));
    CHECK_CUDA(cudaMalloc(&d_ptrA, sizeof(ElementA*) * n_problems));
    CHECK_CUDA(cudaMalloc(&d_ptrB, sizeof(ElementB*) * n_problems));
    CHECK_CUDA(cudaMalloc(&d_ptrC, sizeof(ElementC*) * n_problems));
    CHECK_CUDA(cudaMalloc(&d_lda, sizeof(int64_t) * n_problems));
    CHECK_CUDA(cudaMalloc(&d_ldb, sizeof(int64_t) * n_problems));
    CHECK_CUDA(cudaMalloc(&d_ldc, sizeof(int64_t) * n_problems));
    CHECK_CUDA(cudaMemcpy(d_problem_sizes, problem_sizes.data(), ps_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptrA, h_ptrA.data(), sizeof(ElementA*)*n_problems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptrB, h_ptrB.data(), sizeof(ElementB*)*n_problems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptrC, h_ptrC.data(), sizeof(ElementC*)*n_problems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lda, h_lda.data(), sizeof(int64_t)*n_problems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldb, h_ldb.data(), sizeof(int64_t)*n_problems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ldc, h_ldc.data(), sizeof(int64_t)*n_problems, cudaMemcpyHostToDevice));

    // Pick threadblock count: CUTLASS docs say >= active SMs works well.
    int dev = 0; cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    int threadblock_count = GemmGrouped::sufficient(problem_sizes.data(), n_problems);
    if (threadblock_count == 0) threadblock_count = prop.multiProcessorCount;

    typename GemmGrouped::EpilogueOutputOp::Params epilogue(ElementAcc(1.0f), ElementAcc(0.0f));
    typename GemmGrouped::Arguments args(
        d_problem_sizes, n_problems, threadblock_count, epilogue,
        d_ptrA, d_ptrB, d_ptrC, d_ptrC,  // D == C (in-place; beta=0 so C is ignored)
        d_lda, d_ldb, d_ldc, d_ldc,
        problem_sizes.data());  // host copy used for heuristics

    GemmGrouped gemm;
    size_t ws = gemm.get_workspace_size(args);
    uint8_t* d_ws = nullptr;
    if (ws) CHECK_CUDA(cudaMalloc(&d_ws, ws));
    auto st = gemm.initialize(args, d_ws);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS initialize failed: %d\n", (int)st);
        std::exit(1);
    }

    cudaEvent_t a, b; CHECK_CUDA(cudaEventCreate(&a)); CHECK_CUDA(cudaEventCreate(&b));
    for (int i = 0; i < warmups; i++) CHECK_CUDA((cudaError_t)gemm.run());
    CHECK_CUDA(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) {
        auto rs = gemm.run();
        if (rs != cutlass::Status::kSuccess) {
            std::fprintf(stderr, "CUTLASS run failed: %d\n", (int)rs);
            std::exit(1);
        }
    }
    CHECK_CUDA(cudaEventRecord(b));
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    CHECK_CUDA(cudaEventDestroy(a)); CHECK_CUDA(cudaEventDestroy(b));

    cudaFree(d_problem_sizes);
    cudaFree(d_ptrA); cudaFree(d_ptrB); cudaFree(d_ptrC);
    cudaFree(d_lda); cudaFree(d_ldb); cudaFree(d_ldc);
    if (d_ws) cudaFree(d_ws);
    return ms / iters;
}

// ---------------------------------------------------------------------------
// Verify grouped output == sequential cuBLAS output (per-element max abs diff).
// ---------------------------------------------------------------------------
static bool verify(const __half* a, const __half* b, size_t n, float tol) {
    std::vector<uint16_t> ha(n), hb(n);
    CHECK_CUDA(cudaMemcpy(ha.data(), a, n*2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hb.data(), b, n*2, cudaMemcpyDeviceToHost));
    float max_abs = 0, max_rel = 0;
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        float x = h2f(ha[i]), y = h2f(hb[i]);
        float da = std::fabs(x - y);
        float dr = da / (std::fabs(x) + 1e-6f);
        max_abs = std::max(max_abs, da);
        max_rel = std::max(max_rel, dr);
        if (da > tol && dr > 1e-2f) bad++;
    }
    std::printf("  verify: max_abs=%.4f  max_rel=%.4f  bad=%d/%zu\n", max_abs, max_rel, bad, n);
    return bad == 0;
}

int main(int argc, char** argv) {
    std::string dist = (argc > 1) ? argv[1] : "uniform";
    int T = (argc > 2) ? std::atoi(argv[2]) : 2048;

    // MoE layer shape: match our Qwen-sim setup
    const int K = 4, E = 64;
    const int d = 2048, d_ffn = 8192;
    const int TK = T * K;

    std::printf("=== Grouped GEMM test (%s) ===\n", dist.c_str());
    std::printf("T=%d K=%d E=%d d=%d d_ffn=%d TK=%d\n\n", T, K, E, d, d_ffn, TK);

    // per-expert counts and offsets
    auto M = make_counts(E, TK, dist, 7);
    std::vector<int> off(E, 0);
    for (int e = 1; e < E; e++) off[e] = off[e-1] + M[e-1];
    int max_m = *std::max_element(M.begin(), M.end());
    int min_m = *std::min_element(M.begin(), M.end());
    std::printf("per-expert M: min=%d max=%d mean=%d (imbalance %.1fx)\n\n",
                min_m, max_m, TK/E, (float)max_m / (TK/E));

    // Allocate contiguous buffers
    __half *d_packed, *d_W1, *d_mid_cublas, *d_mid_cutlass;
    CHECK_CUDA(cudaMalloc(&d_packed,      (size_t)TK * d * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_W1,          (size_t)E * d * d_ffn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_mid_cublas,  (size_t)TK * d_ffn * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_mid_cutlass, (size_t)TK * d_ffn * sizeof(__half)));

    // Fill with small random values so output doesn't overflow fp16
    std::vector<uint16_t> h_buf(std::max((size_t)TK*d, (size_t)E*d*d_ffn));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(-0.02f, 0.02f);
    for (size_t i = 0; i < (size_t)TK*d; i++) h_buf[i] = f2h(uni(rng));
    CHECK_CUDA(cudaMemcpy(d_packed, h_buf.data(), (size_t)TK*d*2, cudaMemcpyHostToDevice));
    for (size_t i = 0; i < (size_t)E*d*d_ffn; i++) h_buf[i] = f2h(uni(rng));
    CHECK_CUDA(cudaMemcpy(d_W1, h_buf.data(), (size_t)E*d*d_ffn*2, cudaMemcpyHostToDevice));

    cublasHandle_t cublas; CHECK_CUBLAS(cublasCreate(&cublas));

    const int WARMUPS = 3, ITERS = 10;

    CHECK_CUDA(cudaMemset(d_mid_cublas, 0, (size_t)TK*d_ffn*2));
    float t_seq = run_sequential_cublas(cublas, d_packed, d_W1, d_mid_cublas,
                                        M, off, d, d_ffn, WARMUPS, ITERS);
    std::printf("[seq cuBLAS]  avg %.3f ms  (%d calls/pass)\n", t_seq, E);

    CHECK_CUDA(cudaMemset(d_mid_cutlass, 0, (size_t)TK*d_ffn*2));
    float t_grp = run_grouped_cutlass(d_packed, d_W1, d_mid_cutlass,
                                      M, off, d, d_ffn, WARMUPS, ITERS);
    std::printf("[CUTLASS grp] avg %.3f ms  (1 launch)\n", t_grp);
    std::printf("              speedup: %.2fx\n\n", t_seq / t_grp);

    bool ok = verify(d_mid_cublas, d_mid_cutlass, (size_t)TK*d_ffn, 0.1f);
    std::printf("%s\n", ok ? "PASS" : "FAIL");

    CHECK_CUBLAS(cublasDestroy(cublas));
    cudaFree(d_packed); cudaFree(d_W1); cudaFree(d_mid_cublas); cudaFree(d_mid_cutlass);
    return ok ? 0 : 1;
}
