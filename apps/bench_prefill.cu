// apps/bench_prefill.cu
//
// Multi-round Expert Parallelism benchmark with proper all-to-all-v dispatch.
//
// Pipeline per round (per GPU rank r, with N total ranks):
//   1. Local scatter on T_local = T/N tokens over all E experts
//   2. AllGather expert_count so every GPU knows everyone's per-expert counts
//   3. AllToAll-V dispatch: GPU g sends to rank r the slice of bufs.packed
//      containing tokens for r's experts [r*E/N, (r+1)*E/N)
//   4. FFN on d_dispatch_recv: per (sender, local_expert) cublasHgemm
//   5. AllToAll-V combine: each GPU sends FFN'd outputs back to the GPU that
//      originally sent the tokens
//
// Input is split along the token dimension: GPU r owns rows [r*T/N, (r+1)*T/N]
// of the embeddings/assignments/weights. This matches sequence-parallel + EP
// layouts used by Megablocks / DeepSpeed-MoE / Tutel.
//
// Usage:
//   ./build/bin/bench_prefill --prefix=syn_uniform_T2048_N32 --strategy=warp --n-gpus=2

#include "moe/cpu_reference.hpp"
#include "moe/fp16_host.hpp"
#include "moe/npy_io.hpp"
#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <nccl.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <numeric>
#include <algorithm>
#include <cmath>

// ── Error checks ─────────────────────────────────────────────────────────────

#define NCCL_CHECK(call)                                                        \
    do {                                                                        \
        ncclResult_t _r = (call);                                               \
        if (_r != ncclSuccess) {                                                \
            std::fprintf(stderr, "NCCL error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, ncclGetErrorString(_r));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _st = (call);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n",                   \
                         __FILE__, __LINE__, (int)_st);                        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ── CUTLASS Grouped GEMM (sm_80, fp16/fp16/fp16, RowMajor) ──────────────────
// Tile 128×256×32 / warp 64×64 — best zipf result on A100 retest (1.22×).
// Used optionally for the W1 and W2 FFN matmuls when --ffn=grouped.

using _GG_ElA   = cutlass::half_t;
using _GG_ElB   = cutlass::half_t;
using _GG_ElC   = cutlass::half_t;
using _GG_ElAcc = float;

using GroupedGemm_t = cutlass::gemm::device::GemmGrouped<
    typename cutlass::gemm::kernel::DefaultGemmGrouped<
        _GG_ElA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
        _GG_ElB, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
        _GG_ElC, cutlass::layout::RowMajor,
        _GG_ElAcc,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 256, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<_GG_ElC, 8, _GG_ElAcc, _GG_ElAcc>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4>::GemmKernel>;

struct GroupedGemmCtx {
    // Device-side scratch (max_problems-sized; refilled per call).
    cutlass::gemm::GemmCoord* d_problem_sizes = nullptr;
    _GG_ElA** d_ptrA = nullptr;
    _GG_ElB** d_ptrB = nullptr;
    _GG_ElC** d_ptrC = nullptr;
    int64_t* d_lda = nullptr;
    int64_t* d_ldb = nullptr;
    int64_t* d_ldc = nullptr;
    uint8_t* d_workspace = nullptr;
    std::size_t workspace_bytes = 0;

    // Host scratch (rebuilt per FFN stage; size <= max_problems).
    std::vector<cutlass::gemm::GemmCoord> h_problem_sizes;
    std::vector<_GG_ElA*> h_ptrA;
    std::vector<_GG_ElB*> h_ptrB;
    std::vector<_GG_ElC*> h_ptrC;
    std::vector<int64_t> h_lda, h_ldb, h_ldc;

    int max_problems = 0;
    int sm_count     = 108;  // A100 default; refresh from cudaGetDeviceProperties
};

static void gg_init(GroupedGemmCtx& ctx, int max_problems) {
    ctx.max_problems = max_problems;
    cudaDeviceProp prop;
    int dev = 0;
    MOE_CUDA_CHECK(cudaGetDevice(&dev));
    MOE_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    ctx.sm_count = prop.multiProcessorCount;

    MOE_CUDA_CHECK(cudaMalloc(&ctx.d_problem_sizes,
        max_problems * sizeof(cutlass::gemm::GemmCoord)));
    MOE_CUDA_CHECK(cudaMalloc(&ctx.d_ptrA, max_problems * sizeof(_GG_ElA*)));
    MOE_CUDA_CHECK(cudaMalloc(&ctx.d_ptrB, max_problems * sizeof(_GG_ElB*)));
    MOE_CUDA_CHECK(cudaMalloc(&ctx.d_ptrC, max_problems * sizeof(_GG_ElC*)));
    MOE_CUDA_CHECK(cudaMalloc(&ctx.d_lda, max_problems * sizeof(int64_t)));
    MOE_CUDA_CHECK(cudaMalloc(&ctx.d_ldb, max_problems * sizeof(int64_t)));
    MOE_CUDA_CHECK(cudaMalloc(&ctx.d_ldc, max_problems * sizeof(int64_t)));

    ctx.h_problem_sizes.reserve(max_problems);
    ctx.h_ptrA.reserve(max_problems);
    ctx.h_ptrB.reserve(max_problems);
    ctx.h_ptrC.reserve(max_problems);
    ctx.h_lda.reserve(max_problems);
    ctx.h_ldb.reserve(max_problems);
    ctx.h_ldc.reserve(max_problems);

    // Workspace: estimate from max-shape problem; reuse across calls.
    // We over-allocate via initialize() with a max-size dummy below.
    ctx.workspace_bytes = 0;
    ctx.d_workspace = nullptr;
}

static void gg_destroy(GroupedGemmCtx& ctx) {
    cudaFree(ctx.d_problem_sizes);
    cudaFree(ctx.d_ptrA); cudaFree(ctx.d_ptrB); cudaFree(ctx.d_ptrC);
    cudaFree(ctx.d_lda);  cudaFree(ctx.d_ldb);  cudaFree(ctx.d_ldc);
    if (ctx.d_workspace) cudaFree(ctx.d_workspace);
}

// Issue one grouped-GEMM call onto `stream`. Caller has already populated
// ctx.h_problem_sizes / h_ptrA/B/C / h_lda/b/c.
static void gg_run(GroupedGemmCtx& ctx, cudaStream_t stream) {
    int n = (int)ctx.h_problem_sizes.size();
    if (n == 0) return;

    MOE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_problem_sizes, ctx.h_problem_sizes.data(),
        n * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream));
    MOE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_ptrA, ctx.h_ptrA.data(),
        n * sizeof(_GG_ElA*), cudaMemcpyHostToDevice, stream));
    MOE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_ptrB, ctx.h_ptrB.data(),
        n * sizeof(_GG_ElB*), cudaMemcpyHostToDevice, stream));
    MOE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_ptrC, ctx.h_ptrC.data(),
        n * sizeof(_GG_ElC*), cudaMemcpyHostToDevice, stream));
    MOE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_lda, ctx.h_lda.data(),
        n * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    MOE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_ldb, ctx.h_ldb.data(),
        n * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    MOE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_ldc, ctx.h_ldc.data(),
        n * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    int tb_count = GroupedGemm_t::sufficient(ctx.h_problem_sizes.data(), n);
    if (tb_count == 0) tb_count = ctx.sm_count;

    typename GroupedGemm_t::EpilogueOutputOp::Params epilogue(
        _GG_ElAcc(1.0f), _GG_ElAcc(0.0f));
    typename GroupedGemm_t::Arguments args(
        ctx.d_problem_sizes, n, tb_count, epilogue,
        ctx.d_ptrA, ctx.d_ptrB, ctx.d_ptrC, ctx.d_ptrC,
        ctx.d_lda,  ctx.d_ldb,  ctx.d_ldc, ctx.d_ldc,
        ctx.h_problem_sizes.data());

    GroupedGemm_t gemm;
    std::size_t ws = gemm.get_workspace_size(args);
    if (ws > ctx.workspace_bytes) {
        if (ctx.d_workspace) cudaFree(ctx.d_workspace);
        MOE_CUDA_CHECK(cudaMalloc(&ctx.d_workspace, ws));
        ctx.workspace_bytes = ws;
    }
    auto st = gemm.initialize(args, ctx.d_workspace, stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "GroupedGemm initialize failed: %d\n", (int)st);
        std::exit(1);
    }
    auto rs = gemm.run(stream);
    if (rs != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "GroupedGemm run failed: %d\n", (int)rs);
        std::exit(1);
    }
}

// ── L2 prefetch kernel ────────────────────────────────────────────────────────
// Issues one prefetch.global.L2 per 128-byte cache line (64 fp16 elements).
// Launched on the compute stream so the hardware can fetch while the current
// GEMM runs, hiding the cold-miss latency for the next expert's weight tile.

__global__ void prefetch_l2_kernel(const __half* __restrict__ ptr, std::size_t n_fp16) {
    std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i * 64 < n_fp16; i += stride)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr + i * 64) : "memory");
}

static void prefetch_weights_l2(const __half* ptr, std::size_t n_fp16, cudaStream_t s) {
    if (n_fp16 == 0) return;
    int threads = 256;
    int blocks  = (int)std::min((n_fp16 + 63) / 64 / threads + 1, (std::size_t)32);
    prefetch_l2_kernel<<<blocks, threads, 0, s>>>(ptr, n_fp16);
}

// ── Barrier for N threads ─────────────────────────────────────────────────────

struct BarrierN {
    std::mutex              mtx;
    std::condition_variable cv;
    int count      = 0;
    int generation = 0;
    int n;

    explicit BarrierN(int n_threads) : n(n_threads) {}

    void wait() {
        std::unique_lock<std::mutex> lk(mtx);
        int gen = generation;
        if (++count == n) {
            count = 0;
            ++generation;
            cv.notify_all();
        } else {
            cv.wait(lk, [&]{ return generation != gen; });
        }
    }
};

// ── Per-round result ──────────────────────────────────────────────────────────

struct RoundResult {
    float scatter_ms  = 0.f;
    float dispatch_ms = 0.f;
    float ffn_ms      = 0.f;
    float combine_ms  = 0.f;
    float total_ms    = 0.f;
};

struct PrefillResult {
    std::vector<RoundResult> rounds;
    float  total_ms     = 0.f;
    double local_sum_sq = 0.0;  // sum of squares of d_combine_recv from round 0
    long long n_elts    = 0;    // number of fp16 elements summed (sanity check)
};

// ── CLI config ────────────────────────────────────────────────────────────────

struct Config {
    std::string prefix    = "";
    std::string data_dir  = "results";
    std::string strategy  = "sort";
    int         E_cli     = 64;
    int         n_gpus    = 0;
    int         warmup    = 3;
    int         repeats   = 9;   // odd, larger N → stabler median against cloud noise
    int         overlap   = 0;   // 0 = single-stream serial, 1 = 2-stream comm/compute overlap
    int         chunks    = 1;   // when overlap=1: 1 = unchunked, K>1 = K-chunk dispatch (1F1B-style)
    std::string ffn       = "cublas";  // cublas (per-expert hgemm loop) | grouped (CUTLASS GroupedGemm)
    int         prefetch  = 0;        // 1 = L2-prefetch next expert's weights before current GEMM
};

Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (a.rfind("--", 0) != 0 || eq == std::string::npos) {
            std::fprintf(stderr, "bad arg '%s'\n", argv[i]); std::exit(1);
        }
        std::string k = a.substr(2, eq - 2);
        std::string v = a.substr(eq + 1);
        if      (k == "prefix")   c.prefix   = v;
        else if (k == "data-dir") c.data_dir = v;
        else if (k == "strategy") c.strategy = v;
        else if (k == "E")        c.E_cli    = std::atoi(v.c_str());
        else if (k == "n-gpus")   c.n_gpus   = std::atoi(v.c_str());
        else if (k == "warmup")   c.warmup   = std::atoi(v.c_str());
        else if (k == "repeats")  c.repeats  = std::atoi(v.c_str());
        else if (k == "overlap")  c.overlap  = std::atoi(v.c_str());
        else if (k == "chunks")   c.chunks   = std::atoi(v.c_str());
        else if (k == "ffn")      c.ffn      = v;
        else if (k == "prefetch") c.prefetch = std::atoi(v.c_str());
        else { std::fprintf(stderr, "unknown flag --%s\n", k.c_str()); std::exit(1); }
    }
    if (c.prefix.empty()) {
        std::fprintf(stderr, "usage: --prefix=<name> [--strategy=...] [--n-gpus=N]\n");
        std::exit(1);
    }
    return c;
}

// ── Per-rank worker ───────────────────────────────────────────────────────────

void rank_worker(
    int           rank,
    int           n_ranks,
    ncclComm_t    comm,
    BarrierN&     barrier,
    const Config& cfg,
    PrefillResult* out)
{
    MOE_CUDA_CHECK(cudaSetDevice(rank));

    // ── Load data ─────────────────────────────────────────────────────────────
    auto emb  = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_embeddings.npy");
    auto asgn = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_assignments.npy");
    auto wts  = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_weights.npy");

    const int T_full   = (int)emb.rows;
    const int d_model  = (int)emb.cols;
    const int K        = (int)asgn.cols;
    const int N_rounds = (int)(asgn.rows / T_full);
    if (N_rounds < 1) {
        std::fprintf(stderr, "rank %d: assignments have %zu rows but T_full=%d\n",
                     rank, asgn.rows, T_full);
        std::exit(1);
    }
    if (T_full % n_ranks != 0) {
        std::fprintf(stderr, "T_full=%d not divisible by n_ranks=%d\n", T_full, n_ranks);
        std::exit(1);
    }

    const int T_local  = T_full / n_ranks;
    const int TK_local = T_local * K;
    const int TK_full  = T_full * K;

    const int32_t* h_asgn_all = asgn.as_i32();
    int32_t max_e = -1;
    for (std::size_t i = 0; i < asgn.rows * asgn.cols; ++i)
        max_e = std::max(max_e, h_asgn_all[i]);

    moe::MoEParams p{};
    p.T       = T_local;        // local scatter sees only this rank's tokens
    p.K       = K;
    p.E       = std::max(cfg.E_cli, (int)(max_e + 1));
    p.d_model = d_model;
    p.d_ffn   = d_model * 4;

    if (p.E % n_ranks != 0) {
        std::fprintf(stderr, "E=%d not divisible by n_ranks=%d\n", p.E, n_ranks);
        std::exit(1);
    }
    const int experts_per_gpu = p.E / n_ranks;

    // Convert all weights to fp32 (full T_full*K*N_rounds)
    std::vector<float> wts_f32_all(wts.rows * wts.cols);
    if (wts.dtype == "<f4") {
        std::memcpy(wts_f32_all.data(), wts.data.data(), wts_f32_all.size() * 4);
    } else {
        const __half* h = reinterpret_cast<const __half*>(wts.data.data());
        for (std::size_t i = 0; i < wts_f32_all.size(); ++i)
            wts_f32_all[i] = __half2float(h[i]);
    }

    // ── Device allocations ────────────────────────────────────────────────────
    // Only this rank's slice of inputs (saves memory, no redundant H2D)
    __half*  d_emb_local   = moe::device_alloc<__half>((std::size_t)T_local * d_model);
    int32_t* d_asgn_local  = moe::device_alloc<int32_t>((std::size_t)N_rounds * TK_local);
    float*   d_wts_local   = moe::device_alloc<float>((std::size_t)N_rounds * TK_local);

    // Slice embeddings: rows [rank*T_local, (rank+1)*T_local)
    MOE_CUDA_CHECK(cudaMemcpy(
        d_emb_local,
        emb.as_fp16() + (std::size_t)rank * T_local * d_model,
        (std::size_t)T_local * d_model * 2,
        cudaMemcpyHostToDevice));

    // Slice asgn/wts per round
    for (int i = 0; i < N_rounds; ++i) {
        std::size_t src_off = (std::size_t)i * T_full * K + (std::size_t)rank * TK_local;
        MOE_CUDA_CHECK(cudaMemcpy(
            d_asgn_local + (std::size_t)i * TK_local,
            h_asgn_all + src_off,
            (std::size_t)TK_local * 4,
            cudaMemcpyHostToDevice));
        MOE_CUDA_CHECK(cudaMemcpy(
            d_wts_local + (std::size_t)i * TK_local,
            wts_f32_all.data() + src_off,
            (std::size_t)TK_local * 4,
            cudaMemcpyHostToDevice));
    }

    // Scatter buffers (sized for local scatter: T_local tokens × K)
    moe::MoEBuffers bufs{};
    bufs.packed         = moe::device_alloc<__half>((std::size_t)TK_local * d_model);
    bufs.packed_weights = moe::device_alloc<float>((std::size_t)TK_local);
    bufs.perm           = moe::device_alloc<int32_t>((std::size_t)TK_local);
    bufs.expert_count   = moe::device_alloc<int32_t>((std::size_t)p.E);
    bufs.expert_start   = moe::device_alloc<int32_t>((std::size_t)(p.E + 1));
    bufs.strategy_ws_bytes = moe::make_scatter(cfg.strategy)->workspace_bytes(p);
    MOE_CUDA_CHECK(cudaMalloc(&bufs.strategy_ws, bufs.strategy_ws_bytes));

    // AllGather buffer for expert_count from every rank
    int32_t* d_all_counts = moe::device_alloc<int32_t>((std::size_t)n_ranks * p.E);

    // Dispatch/combine recv buffers. Worst case recv = TK_full (all tokens to
    // this GPU's experts under extreme imbalance). Allocate conservatively.
    __half* d_dispatch_recv = moe::device_alloc<__half>((std::size_t)TK_full * d_model);
    __half* d_combine_recv  = moe::device_alloc<__half>((std::size_t)TK_local * d_model);

    // FFN buffers sized for max possible recv
    __half* d_ffn_tmp    = moe::device_alloc<__half>((std::size_t)TK_full * p.d_ffn);
    __half* d_expert_out = moe::device_alloc<__half>((std::size_t)TK_full * d_model);

    // FFN weights: each GPU runs experts_per_gpu experts
    __half* d_W1 = moe::device_alloc<__half>(
        (std::size_t)experts_per_gpu * d_model * p.d_ffn);
    __half* d_W2 = moe::device_alloc<__half>(
        (std::size_t)experts_per_gpu * p.d_ffn * d_model);
    {
        std::size_t w1_size = (std::size_t)experts_per_gpu * d_model * p.d_ffn;
        std::size_t w2_size = (std::size_t)experts_per_gpu * p.d_ffn * d_model;
        std::vector<uint16_t> h_W1(w1_size), h_W2(w2_size);
        // Seed by global expert id so 1-GPU and N-GPU runs use matching weights
        for (int e_local = 0; e_local < experts_per_gpu; ++e_local) {
            int e_global = rank * experts_per_gpu + e_local;
            moe::fill_random_half(h_W1.data() + (std::size_t)e_local * d_model * p.d_ffn,
                                  (std::size_t)d_model * p.d_ffn, 1000 + e_global);
            moe::fill_random_half(h_W2.data() + (std::size_t)e_local * p.d_ffn * d_model,
                                  (std::size_t)p.d_ffn * d_model, 2000 + e_global);
        }
        MOE_CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), w1_size * 2, cudaMemcpyHostToDevice));
        MOE_CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), w2_size * 2, cudaMemcpyHostToDevice));
    }

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    auto scatter = moe::make_scatter(cfg.strategy);

    // Two streams: compute (scatter + FFN) and comm (AllGather + AllToAll-V).
    // For overlap=0 they're aliased to the same stream so the harness keeps
    // the legacy single-stream behavior for an apples-to-apples baseline.
    cudaStream_t s_cmp, s_com;
    MOE_CUDA_CHECK(cudaStreamCreate(&s_cmp));
    if (cfg.overlap) {
        MOE_CUDA_CHECK(cudaStreamCreate(&s_com));
    } else {
        s_com = s_cmp;
    }
    CUBLAS_CHECK(cublasSetStream(cublas, s_cmp));

    // chunks must divide experts_per_gpu evenly when chunked.
    const int K_CHUNKS = (cfg.overlap && cfg.chunks > 1) ? cfg.chunks : 1;
    if (K_CHUNKS > 1 && experts_per_gpu % K_CHUNKS != 0) {
        std::fprintf(stderr,
            "chunks=%d must divide experts_per_gpu=%d evenly\n",
            K_CHUNKS, experts_per_gpu);
        std::exit(1);
    }
    const int experts_per_chunk = experts_per_gpu / K_CHUNKS;

    // Cross-stream events for handoff. Per-chunk events when chunked, so
    // dispatch chunk c+1 can overlap FFN chunk c on different streams.
    cudaEvent_t ev_scatter_done;
    MOE_CUDA_CHECK(cudaEventCreateWithFlags(&ev_scatter_done, cudaEventDisableTiming));
    std::vector<cudaEvent_t> ev_dispatch_done(K_CHUNKS), ev_ffn_done(K_CHUNKS);
    for (int c = 0; c < K_CHUNKS; ++c) {
        MOE_CUDA_CHECK(cudaEventCreateWithFlags(&ev_dispatch_done[c], cudaEventDisableTiming));
        MOE_CUDA_CHECK(cudaEventCreateWithFlags(&ev_ffn_done[c],      cudaEventDisableTiming));
    }

    __half alpha_h = __float2half(1.f), beta_h = __float2half(0.f);

    // Grouped-GEMM context (only used when --ffn=grouped). Max problems per
    // call = n_ranks * experts_per_gpu (one (sender, local_expert) pair).
    const bool use_grouped_ffn = (cfg.ffn == "grouped");
    GroupedGemmCtx ggW1, ggW2;
    if (use_grouped_ffn) {
        gg_init(ggW1, n_ranks * experts_per_gpu);
        gg_init(ggW2, n_ranks * experts_per_gpu);
    }

    // Host-side scratch (reused across rounds)
    std::vector<int32_t> h_all_counts((std::size_t)n_ranks * p.E);
    std::vector<int32_t> h_send_off_tok(n_ranks + 1);  // token offsets in bufs.packed
    std::vector<int32_t> h_recv_off_tok(n_ranks + 1);  // token offsets in d_dispatch_recv
    // Per-chunk per-rank send/recv counts (only used when K_CHUNKS > 1).
    // Layout: chunk_send_cnt[c * n_ranks + g] = tokens this rank sends to g for chunk c.
    std::vector<int32_t> chunk_send_cnt((std::size_t)K_CHUNKS * n_ranks);
    std::vector<int32_t> chunk_recv_cnt((std::size_t)K_CHUNKS * n_ranks);
    std::vector<int32_t> chunk_send_off((std::size_t)K_CHUNKS * n_ranks);  // offset within sender's bufs.packed
    std::vector<int32_t> chunk_recv_off((std::size_t)K_CHUNKS * n_ranks);  // offset within d_dispatch_recv

    // ── Lambda: run one full round ────────────────────────────────────────────
    // Two paths:
    //   serial   (cfg.overlap=0)  legacy single-stream with per-stage timers
    //                             and inter-stage barriers. Reports per-stage
    //                             breakdown.
    //   overlap  (cfg.overlap=1)  2-stream pipeline. scatter+FFN on s_cmp,
    //                             AllGather+dispatch+combine on s_com. Cross-
    //                             stream events serialize within a round but
    //                             allow cross-round overlap. No per-stage
    //                             timers (would force host syncs that defeat
    //                             pipelining); only round-level wall.
    auto run_round = [&](int round_idx, RoundResult* rr) {
        int32_t* d_asgn = d_asgn_local + (std::size_t)round_idx * TK_local;
        float*   d_wts  = d_wts_local  + (std::size_t)round_idx * TK_local;

        if (cfg.overlap) {
            // ── OVERLAP path (with optional K-chunk 1F1B) ───────────────────
            // Stage 1: scatter on s_cmp
            scatter->run(d_emb_local, d_asgn, d_wts, bufs, p, s_cmp);
            MOE_CUDA_CHECK(cudaEventRecord(ev_scatter_done, s_cmp));

            // Stage 2a (s_com): AllGather + sync (always one shot — small)
            MOE_CUDA_CHECK(cudaStreamWaitEvent(s_com, ev_scatter_done, 0));
            NCCL_CHECK(ncclAllGather(
                bufs.expert_count, d_all_counts, p.E, ncclInt32, comm, s_com));
            MOE_CUDA_CHECK(cudaMemcpyAsync(
                h_all_counts.data(), d_all_counts,
                (std::size_t)n_ranks * p.E * 4, cudaMemcpyDeviceToHost, s_com));
            MOE_CUDA_CHECK(cudaStreamSynchronize(s_com));

            // 2b. Compute global per-pair offsets (used for both unchunked
            //     and chunked path; chunked path further splits these by
            //     experts_per_chunk).
            h_send_off_tok[0] = 0;
            for (int g = 0; g < n_ranks; ++g) {
                int32_t s = 0;
                for (int e = g * experts_per_gpu; e < (g + 1) * experts_per_gpu; ++e)
                    s += h_all_counts[(std::size_t)rank * p.E + e];
                h_send_off_tok[g + 1] = h_send_off_tok[g] + s;
            }
            h_recv_off_tok[0] = 0;
            for (int g = 0; g < n_ranks; ++g) {
                int32_t s = 0;
                for (int e = rank * experts_per_gpu; e < (rank + 1) * experts_per_gpu; ++e)
                    s += h_all_counts[(std::size_t)g * p.E + e];
                h_recv_off_tok[g + 1] = h_recv_off_tok[g] + s;
            }

            // 2c. When chunked, derive per-(chunk, rank) send/recv counts +
            //     offsets. The slice for chunk c sent to rank r covers r's
            //     experts [c*experts_per_chunk, (c+1)*experts_per_chunk).
            if (K_CHUNKS > 1) {
                for (int c = 0; c < K_CHUNKS; ++c) {
                    for (int g = 0; g < n_ranks; ++g) {
                        int32_t scnt = 0, rcnt = 0;
                        // Send: this rank's tokens for g's experts in chunk c
                        int e_lo_send = g * experts_per_gpu + c * experts_per_chunk;
                        int e_hi_send = e_lo_send + experts_per_chunk;
                        for (int e = e_lo_send; e < e_hi_send; ++e)
                            scnt += h_all_counts[(std::size_t)rank * p.E + e];
                        // Recv: g's tokens for our experts in chunk c
                        int e_lo_recv = rank * experts_per_gpu + c * experts_per_chunk;
                        int e_hi_recv = e_lo_recv + experts_per_chunk;
                        for (int e = e_lo_recv; e < e_hi_recv; ++e)
                            rcnt += h_all_counts[(std::size_t)g * p.E + e];
                        chunk_send_cnt[c * n_ranks + g] = scnt;
                        chunk_recv_cnt[c * n_ranks + g] = rcnt;
                    }
                }
                // Compute offsets within each rank's bufs.packed (send) and
                // d_dispatch_recv (recv). For rank r: offset starts at
                // h_send_off_tok[g] (which is the cumulative offset for the
                // entire pre-r block) and within that, chunk c starts after
                // chunks 0..c-1.
                for (int g = 0; g < n_ranks; ++g) {
                    int32_t s_off = h_send_off_tok[g];
                    int32_t r_off = h_recv_off_tok[g];
                    for (int c = 0; c < K_CHUNKS; ++c) {
                        chunk_send_off[c * n_ranks + g] = s_off;
                        chunk_recv_off[c * n_ranks + g] = r_off;
                        s_off += chunk_send_cnt[c * n_ranks + g];
                        r_off += chunk_recv_cnt[c * n_ranks + g];
                    }
                }
            }

            // 2d. Per-chunk dispatch + per-chunk FFN + per-chunk combine.
            //     Unchunked is just K_CHUNKS=1 (single iteration).
            for (int c = 0; c < K_CHUNKS; ++c) {
                // ── Dispatch chunk c on s_com ────────────────────────────
                NCCL_CHECK(ncclGroupStart());
                for (int g = 0; g < n_ranks; ++g) {
                    std::size_t send_cnt, recv_cnt;
                    int32_t send_off, recv_off;
                    if (K_CHUNKS == 1) {
                        send_cnt = (std::size_t)(h_send_off_tok[g + 1] - h_send_off_tok[g]) * d_model;
                        recv_cnt = (std::size_t)(h_recv_off_tok[g + 1] - h_recv_off_tok[g]) * d_model;
                        send_off = h_send_off_tok[g];
                        recv_off = h_recv_off_tok[g];
                    } else {
                        send_cnt = (std::size_t)chunk_send_cnt[c * n_ranks + g] * d_model;
                        recv_cnt = (std::size_t)chunk_recv_cnt[c * n_ranks + g] * d_model;
                        send_off = chunk_send_off[c * n_ranks + g];
                        recv_off = chunk_recv_off[c * n_ranks + g];
                    }
                    if (send_cnt > 0) {
                        NCCL_CHECK(ncclSend(
                            bufs.packed + (std::size_t)send_off * d_model,
                            send_cnt, ncclHalf, g, comm, s_com));
                    }
                    if (recv_cnt > 0) {
                        NCCL_CHECK(ncclRecv(
                            d_dispatch_recv + (std::size_t)recv_off * d_model,
                            recv_cnt, ncclHalf, g, comm, s_com));
                    }
                }
                NCCL_CHECK(ncclGroupEnd());
                MOE_CUDA_CHECK(cudaEventRecord(ev_dispatch_done[c], s_com));

                // ── FFN chunk c on s_cmp ─────────────────────────────────
                MOE_CUDA_CHECK(cudaStreamWaitEvent(s_cmp, ev_dispatch_done[c], 0));
                int e_local_lo = c * experts_per_chunk;
                int e_local_hi = (c + 1) * experts_per_chunk;
                if (K_CHUNKS == 1) { e_local_lo = 0; e_local_hi = experts_per_gpu; }

                // Walk d_dispatch_recv following the same per-(sender, expert)
                // order it was written. Skip the (sender, expert) pairs whose
                // expert is not in this chunk's range.
                if (use_grouped_ffn) {
                    // Build per-(sender, local_expert) problem list for THIS
                    // chunk, then issue a single GroupedGemm for W1 and a
                    // single GroupedGemm for W2.
                    ggW1.h_problem_sizes.clear(); ggW1.h_ptrA.clear();
                    ggW1.h_ptrB.clear(); ggW1.h_ptrC.clear();
                    ggW1.h_lda.clear(); ggW1.h_ldb.clear(); ggW1.h_ldc.clear();
                    ggW2.h_problem_sizes.clear(); ggW2.h_ptrA.clear();
                    ggW2.h_ptrB.clear(); ggW2.h_ptrC.clear();
                    ggW2.h_lda.clear(); ggW2.h_ldb.clear(); ggW2.h_ldc.clear();

                    std::size_t curr_tok = 0;
                    for (int g = 0; g < n_ranks; ++g) {
                        for (int e_local = 0; e_local < experts_per_gpu; ++e_local) {
                            int e_global = rank * experts_per_gpu + e_local;
                            int cnt = h_all_counts[(std::size_t)g * p.E + e_global];
                            if (cnt == 0) continue;
                            if (e_local < e_local_lo || e_local >= e_local_hi) {
                                curr_tok += cnt; continue;
                            }
                            __half* A   = d_dispatch_recv + curr_tok * d_model;
                            __half* B1  = d_W1 + (std::size_t)e_local * d_model * p.d_ffn;
                            __half* B2  = d_W2 + (std::size_t)e_local * p.d_ffn * d_model;
                            __half* mid = d_ffn_tmp    + curr_tok * p.d_ffn;
                            __half* eout= d_expert_out + curr_tok * d_model;

                            // W1: (cnt, d_ffn, d_model) M×N×K, RowMajor
                            //   A = d_dispatch_recv slice [cnt, d_model], lda=d_model
                            //   B = W1[e]              [d_model, d_ffn], ldb=d_ffn
                            //   C = mid                [cnt, d_ffn],     ldc=d_ffn
                            ggW1.h_problem_sizes.emplace_back(cnt, p.d_ffn, d_model);
                            ggW1.h_ptrA.push_back(reinterpret_cast<_GG_ElA*>(A));
                            ggW1.h_ptrB.push_back(reinterpret_cast<_GG_ElB*>(B1));
                            ggW1.h_ptrC.push_back(reinterpret_cast<_GG_ElC*>(mid));
                            ggW1.h_lda.push_back(d_model);
                            ggW1.h_ldb.push_back(p.d_ffn);
                            ggW1.h_ldc.push_back(p.d_ffn);

                            // W2: (cnt, d_model, d_ffn) M×N×K, RowMajor
                            //   A = mid              [cnt, d_ffn],   lda=d_ffn
                            //   B = W2[e]            [d_ffn, d_model], ldb=d_model
                            //   C = eout             [cnt, d_model], ldc=d_model
                            ggW2.h_problem_sizes.emplace_back(cnt, d_model, p.d_ffn);
                            ggW2.h_ptrA.push_back(reinterpret_cast<_GG_ElA*>(mid));
                            ggW2.h_ptrB.push_back(reinterpret_cast<_GG_ElB*>(B2));
                            ggW2.h_ptrC.push_back(reinterpret_cast<_GG_ElC*>(eout));
                            ggW2.h_lda.push_back(p.d_ffn);
                            ggW2.h_ldb.push_back(d_model);
                            ggW2.h_ldc.push_back(d_model);

                            curr_tok += cnt;
                        }
                    }
                    gg_run(ggW1, s_cmp);
                    gg_run(ggW2, s_cmp);
                } else {
                    std::size_t curr_tok = 0;
                    for (int g = 0; g < n_ranks; ++g) {
                        for (int e_local = 0; e_local < experts_per_gpu; ++e_local) {
                            int e_global = rank * experts_per_gpu + e_local;
                            int cnt = h_all_counts[(std::size_t)g * p.E + e_global];
                            if (cnt == 0) continue;
                            if (e_local < e_local_lo || e_local >= e_local_hi) {
                                curr_tok += cnt; continue;
                            }
                            // Prefetch next expert's weights before this GEMM so the
                            // hardware can fetch while Tensor Cores run current expert.
                            if (cfg.prefetch && g == 0 && e_local + 1 < experts_per_gpu) {
                                std::size_t w_elts = (std::size_t)d_model * p.d_ffn;
                                prefetch_weights_l2(d_W1 + (e_local + 1) * w_elts, w_elts, s_cmp);
                                prefetch_weights_l2(d_W2 + (e_local + 1) * w_elts, w_elts, s_cmp);
                            }
                            const __half* A  = d_dispatch_recv + curr_tok * d_model;
                            const __half* B1 = d_W1 + (std::size_t)e_local * d_model * p.d_ffn;
                            const __half* B2 = d_W2 + (std::size_t)e_local * p.d_ffn * d_model;
                            __half* mid      = d_ffn_tmp    + curr_tok * p.d_ffn;
                            __half* eout     = d_expert_out + curr_tok * d_model;

                            CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                p.d_ffn, cnt, d_model, &alpha_h,
                                B1, p.d_ffn, A, d_model,
                                &beta_h, mid, p.d_ffn));
                            CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                d_model, cnt, p.d_ffn, &alpha_h,
                                B2, d_model, mid, p.d_ffn,
                                &beta_h, eout, d_model));

                            curr_tok += cnt;
                        }
                    }
                }
                MOE_CUDA_CHECK(cudaEventRecord(ev_ffn_done[c], s_cmp));

                // ── Combine chunk c on s_com ─────────────────────────────
                MOE_CUDA_CHECK(cudaStreamWaitEvent(s_com, ev_ffn_done[c], 0));
                NCCL_CHECK(ncclGroupStart());
                for (int g = 0; g < n_ranks; ++g) {
                    std::size_t send_cnt, recv_cnt;
                    int32_t send_off_back, recv_off_back;
                    if (K_CHUNKS == 1) {
                        send_cnt = (std::size_t)(h_recv_off_tok[g + 1] - h_recv_off_tok[g]) * d_model;
                        recv_cnt = (std::size_t)(h_send_off_tok[g + 1] - h_send_off_tok[g]) * d_model;
                        send_off_back = h_recv_off_tok[g];
                        recv_off_back = h_send_off_tok[g];
                    } else {
                        send_cnt = (std::size_t)chunk_recv_cnt[c * n_ranks + g] * d_model;
                        recv_cnt = (std::size_t)chunk_send_cnt[c * n_ranks + g] * d_model;
                        send_off_back = chunk_recv_off[c * n_ranks + g];
                        recv_off_back = chunk_send_off[c * n_ranks + g];
                    }
                    if (send_cnt > 0) {
                        NCCL_CHECK(ncclSend(
                            d_expert_out + (std::size_t)send_off_back * d_model,
                            send_cnt, ncclHalf, g, comm, s_com));
                    }
                    if (recv_cnt > 0) {
                        NCCL_CHECK(ncclRecv(
                            d_combine_recv + (std::size_t)recv_off_back * d_model,
                            recv_cnt, ncclHalf, g, comm, s_com));
                    }
                }
                NCCL_CHECK(ncclGroupEnd());
            }
            // No sync here — outer rep loop syncs once at the end.
            return;
        }

        // ── SERIAL path (legacy, for A/B comparison) ────────────────────────
        moe::CudaTimer ts, td, tf, tc;

        barrier.wait();
        ts.start(s_cmp);
        scatter->run(d_emb_local, d_asgn, d_wts, bufs, p, s_cmp);
        rr->scatter_ms = ts.stop_ms(s_cmp);

        barrier.wait();
        td.start(s_cmp);
        NCCL_CHECK(ncclAllGather(
            bufs.expert_count, d_all_counts, p.E, ncclInt32, comm, s_cmp));
        MOE_CUDA_CHECK(cudaMemcpyAsync(
            h_all_counts.data(), d_all_counts,
            (std::size_t)n_ranks * p.E * 4, cudaMemcpyDeviceToHost, s_cmp));
        MOE_CUDA_CHECK(cudaStreamSynchronize(s_cmp));

        h_send_off_tok[0] = 0;
        for (int g = 0; g < n_ranks; ++g) {
            int32_t s = 0;
            for (int e = g * experts_per_gpu; e < (g + 1) * experts_per_gpu; ++e)
                s += h_all_counts[(std::size_t)rank * p.E + e];
            h_send_off_tok[g + 1] = h_send_off_tok[g] + s;
        }
        h_recv_off_tok[0] = 0;
        for (int g = 0; g < n_ranks; ++g) {
            int32_t s = 0;
            for (int e = rank * experts_per_gpu; e < (rank + 1) * experts_per_gpu; ++e)
                s += h_all_counts[(std::size_t)g * p.E + e];
            h_recv_off_tok[g + 1] = h_recv_off_tok[g] + s;
        }

        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < n_ranks; ++g) {
            std::size_t send_cnt =
                (std::size_t)(h_send_off_tok[g + 1] - h_send_off_tok[g]) * d_model;
            std::size_t recv_cnt =
                (std::size_t)(h_recv_off_tok[g + 1] - h_recv_off_tok[g]) * d_model;
            if (send_cnt > 0) {
                NCCL_CHECK(ncclSend(
                    bufs.packed + (std::size_t)h_send_off_tok[g] * d_model,
                    send_cnt, ncclHalf, g, comm, s_cmp));
            }
            if (recv_cnt > 0) {
                NCCL_CHECK(ncclRecv(
                    d_dispatch_recv + (std::size_t)h_recv_off_tok[g] * d_model,
                    recv_cnt, ncclHalf, g, comm, s_cmp));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(s_cmp));
        rr->dispatch_ms = td.stop_ms(s_cmp);

        tf.start(s_cmp);
        if (use_grouped_ffn) {
            ggW1.h_problem_sizes.clear(); ggW1.h_ptrA.clear();
            ggW1.h_ptrB.clear(); ggW1.h_ptrC.clear();
            ggW1.h_lda.clear(); ggW1.h_ldb.clear(); ggW1.h_ldc.clear();
            ggW2.h_problem_sizes.clear(); ggW2.h_ptrA.clear();
            ggW2.h_ptrB.clear(); ggW2.h_ptrC.clear();
            ggW2.h_lda.clear(); ggW2.h_ldb.clear(); ggW2.h_ldc.clear();

            std::size_t curr_tok = 0;
            for (int g = 0; g < n_ranks; ++g) {
                for (int e_local = 0; e_local < experts_per_gpu; ++e_local) {
                    int e_global = rank * experts_per_gpu + e_local;
                    int cnt = h_all_counts[(std::size_t)g * p.E + e_global];
                    if (cnt == 0) continue;
                    __half* A   = d_dispatch_recv + curr_tok * d_model;
                    __half* B1  = d_W1 + (std::size_t)e_local * d_model * p.d_ffn;
                    __half* B2  = d_W2 + (std::size_t)e_local * p.d_ffn * d_model;
                    __half* mid = d_ffn_tmp    + curr_tok * p.d_ffn;
                    __half* eout= d_expert_out + curr_tok * d_model;

                    ggW1.h_problem_sizes.emplace_back(cnt, p.d_ffn, d_model);
                    ggW1.h_ptrA.push_back(reinterpret_cast<_GG_ElA*>(A));
                    ggW1.h_ptrB.push_back(reinterpret_cast<_GG_ElB*>(B1));
                    ggW1.h_ptrC.push_back(reinterpret_cast<_GG_ElC*>(mid));
                    ggW1.h_lda.push_back(d_model);
                    ggW1.h_ldb.push_back(p.d_ffn);
                    ggW1.h_ldc.push_back(p.d_ffn);

                    ggW2.h_problem_sizes.emplace_back(cnt, d_model, p.d_ffn);
                    ggW2.h_ptrA.push_back(reinterpret_cast<_GG_ElA*>(mid));
                    ggW2.h_ptrB.push_back(reinterpret_cast<_GG_ElB*>(B2));
                    ggW2.h_ptrC.push_back(reinterpret_cast<_GG_ElC*>(eout));
                    ggW2.h_lda.push_back(p.d_ffn);
                    ggW2.h_ldb.push_back(d_model);
                    ggW2.h_ldc.push_back(d_model);

                    curr_tok += cnt;
                }
            }
            gg_run(ggW1, s_cmp);
            gg_run(ggW2, s_cmp);
        } else {
            std::size_t curr_tok = 0;
            for (int g = 0; g < n_ranks; ++g) {
                for (int e_local = 0; e_local < experts_per_gpu; ++e_local) {
                    int e_global = rank * experts_per_gpu + e_local;
                    int cnt = h_all_counts[(std::size_t)g * p.E + e_global];
                    if (cnt == 0) continue;

                    if (cfg.prefetch && g == 0 && e_local + 1 < experts_per_gpu) {
                        std::size_t w_elts = (std::size_t)d_model * p.d_ffn;
                        prefetch_weights_l2(d_W1 + (e_local + 1) * w_elts, w_elts, s_cmp);
                        prefetch_weights_l2(d_W2 + (e_local + 1) * w_elts, w_elts, s_cmp);
                    }
                    const __half* A  = d_dispatch_recv + curr_tok * d_model;
                    const __half* B1 = d_W1 + (std::size_t)e_local * d_model * p.d_ffn;
                    const __half* B2 = d_W2 + (std::size_t)e_local * p.d_ffn * d_model;
                    __half* mid      = d_ffn_tmp    + curr_tok * p.d_ffn;
                    __half* eout     = d_expert_out + curr_tok * d_model;

                    CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        p.d_ffn, cnt, d_model, &alpha_h,
                        B1, p.d_ffn, A, d_model,
                        &beta_h, mid, p.d_ffn));
                    CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        d_model, cnt, p.d_ffn, &alpha_h,
                        B2, d_model, mid, p.d_ffn,
                        &beta_h, eout, d_model));

                    curr_tok += cnt;
                }
            }
        }
        rr->ffn_ms = tf.stop_ms(s_cmp);

        barrier.wait();
        tc.start(s_cmp);
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < n_ranks; ++g) {
            std::size_t send_cnt =
                (std::size_t)(h_recv_off_tok[g + 1] - h_recv_off_tok[g]) * d_model;
            std::size_t recv_cnt =
                (std::size_t)(h_send_off_tok[g + 1] - h_send_off_tok[g]) * d_model;
            if (send_cnt > 0) {
                NCCL_CHECK(ncclSend(
                    d_expert_out + (std::size_t)h_recv_off_tok[g] * d_model,
                    send_cnt, ncclHalf, g, comm, s_cmp));
            }
            if (recv_cnt > 0) {
                NCCL_CHECK(ncclRecv(
                    d_combine_recv + (std::size_t)h_send_off_tok[g] * d_model,
                    recv_cnt, ncclHalf, g, comm, s_cmp));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(s_cmp));
        rr->combine_ms = tc.stop_ms(s_cmp);

        rr->total_ms = rr->scatter_ms + rr->dispatch_ms + rr->ffn_ms + rr->combine_ms;
        barrier.wait();
    };

    // ── Warmup ────────────────────────────────────────────────────────────────
    for (int w = 0; w < cfg.warmup; ++w) {
        RoundResult dummy;
        run_round(w % N_rounds, &dummy);
    }

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<float> total_times(cfg.repeats);
    std::vector<std::vector<RoundResult>> all_round_results(cfg.repeats);

    for (int rep = 0; rep < cfg.repeats; ++rep) {
        all_round_results[rep].resize(N_rounds);
        moe::CudaTimer t_total;

        barrier.wait();
        t_total.start(s_cmp);

        for (int r = 0; r < N_rounds; ++r) {
            run_round(r, &all_round_results[rep][r]);
        }

        // In overlap mode, the inner loop never syncs s_com; combine work for
        // the last rounds is still in flight. Sync both streams now so the
        // total timer captures the full pipeline drain.
        if (cfg.overlap) {
            MOE_CUDA_CHECK(cudaStreamSynchronize(s_com));
            MOE_CUDA_CHECK(cudaStreamSynchronize(s_cmp));
        }
        total_times[rep] = t_total.stop_ms(s_cmp);
    }

    // ── Median over repetitions ──────────────────────────────────────────────
    std::vector<int> order(cfg.repeats);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return total_times[a] < total_times[b]; });
    int median_idx = order[cfg.repeats / 2];

    out->total_ms = total_times[median_idx];
    out->rounds   = all_round_results[median_idx];

    // ── L2 checksum: re-run round 0 deterministically and sum |x|² over
    //    d_combine_recv on host. Aggregating across ranks in main() gives the
    //    global L2 norm — should match between 1-GPU and N-GPU runs. ────────
    {
        RoundResult dummy;
        run_round(0, &dummy);
        std::size_t n_elts = (std::size_t)TK_local * d_model;
        std::vector<uint16_t> h_buf(n_elts);
        MOE_CUDA_CHECK(cudaMemcpy(h_buf.data(), d_combine_recv,
            n_elts * 2, cudaMemcpyDeviceToHost));
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < n_elts; ++i) {
            float v = moe::h2f(h_buf[i]);
            sum_sq += (double)v * v;
        }
        out->local_sum_sq = sum_sq;
        out->n_elts       = (long long)n_elts;
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cublasDestroy(cublas);
    cudaFree(d_emb_local); cudaFree(d_asgn_local); cudaFree(d_wts_local);
    cudaFree(bufs.packed); cudaFree(bufs.packed_weights); cudaFree(bufs.perm);
    cudaFree(bufs.expert_count); cudaFree(bufs.expert_start);
    cudaFree(bufs.strategy_ws);
    cudaFree(d_all_counts);
    cudaFree(d_dispatch_recv); cudaFree(d_combine_recv);
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_ffn_tmp); cudaFree(d_expert_out);
    cudaEventDestroy(ev_scatter_done);
    for (int c = 0; c < K_CHUNKS; ++c) {
        cudaEventDestroy(ev_dispatch_done[c]);
        cudaEventDestroy(ev_ffn_done[c]);
    }
    if (use_grouped_ffn) {
        gg_destroy(ggW1);
        gg_destroy(ggW2);
    }
    cudaStreamDestroy(s_cmp);
    if (cfg.overlap) cudaStreamDestroy(s_com);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Config cfg = parse(argc, argv);

    int n_available = 0;
    MOE_CUDA_CHECK(cudaGetDeviceCount(&n_available));

    int n_ranks = cfg.n_gpus > 0 ? cfg.n_gpus : n_available;
    if (n_ranks < 1) {
        std::fprintf(stderr, "Need at least 1 GPU, found %d\n", n_available);
        return 1;
    }
    if (n_ranks > n_available) {
        std::fprintf(stderr, "Requested %d GPUs but only %d available\n",
                     n_ranks, n_available);
        return 1;
    }

    std::printf("\n== bench_prefill  strategy=%s  prefix=%s  n_gpus=%d  overlap=%d  chunks=%d  ffn=%s  prefetch=%d ==\n\n",
                cfg.strategy.c_str(), cfg.prefix.c_str(), n_ranks, cfg.overlap,
                cfg.chunks, cfg.ffn.c_str(), cfg.prefetch);

    // ── NCCL init ─────────────────────────────────────────────────────────────
    std::vector<ncclComm_t> comms(n_ranks);
    std::vector<int> dev_ids(n_ranks);
    for (int i = 0; i < n_ranks; ++i) dev_ids[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), n_ranks, dev_ids.data()));

    // ── Launch per-rank threads ──────────────────────────────────────────────
    BarrierN barrier(n_ranks);
    std::vector<PrefillResult> results(n_ranks);
    std::vector<std::thread> threads;

    for (int r = 0; r < n_ranks; ++r) {
        threads.emplace_back(rank_worker, r, n_ranks, comms[r],
                             std::ref(barrier), std::cref(cfg), &results[r]);
    }
    for (auto& t : threads) t.join();

    // ── Aggregate and print ──────────────────────────────────────────────────
    int N_rounds = (int)results[0].rounds.size();
    float total_ms = 0;
    for (int r = 0; r < n_ranks; ++r) total_ms += results[r].total_ms;
    total_ms /= n_ranks;

    float avg_scatter = 0, avg_dispatch = 0, avg_ffn = 0, avg_combine = 0;
    for (int r = 0; r < n_ranks; ++r) {
        for (int i = 0; i < N_rounds; ++i) {
            avg_scatter  += results[r].rounds[i].scatter_ms;
            avg_dispatch += results[r].rounds[i].dispatch_ms;
            avg_ffn      += results[r].rounds[i].ffn_ms;
            avg_combine  += results[r].rounds[i].combine_ms;
        }
    }
    float divisor = (float)(n_ranks * N_rounds);
    avg_scatter  /= divisor;
    avg_dispatch /= divisor;
    avg_ffn      /= divisor;
    avg_combine  /= divisor;
    float avg_round = avg_scatter + avg_dispatch + avg_ffn + avg_combine;

    std::printf("  warmup=%d  repeats=%d  rounds=%d  (median total over repeats, avg over %d GPUs)\n\n",
                cfg.warmup, cfg.repeats, N_rounds, n_ranks);

    if (!cfg.overlap) {
        std::printf("  Per-round averages:\n");
        std::printf("    scatter:   %8.3f ms  (%5.1f%%)\n",
                    avg_scatter,  avg_scatter  / avg_round * 100.f);
        std::printf("    dispatch:  %8.3f ms  (%5.1f%%)  [AllGather + AllToAll-V]\n",
                    avg_dispatch, avg_dispatch / avg_round * 100.f);
        std::printf("    FFN:       %8.3f ms  (%5.1f%%)\n",
                    avg_ffn,      avg_ffn      / avg_round * 100.f);
        std::printf("    combine:   %8.3f ms  (%5.1f%%)  [AllToAll-V reverse]\n",
                    avg_combine,  avg_combine  / avg_round * 100.f);
        std::printf("    ─────────────────────────────────\n");
        std::printf("    round:     %8.3f ms\n\n", avg_round);
    } else {
        std::printf("  Per-stage breakdown not measured in overlap mode\n"
                    "  (sub-stage timers would force host syncs that defeat pipelining).\n\n");
    }

    std::printf("  Full prefill (%d rounds):\n", N_rounds);
    std::printf("    total:     %8.3f ms\n", total_ms);
    std::printf("    throughput: %.1f rounds/s\n", N_rounds / (total_ms / 1000.f));
    std::printf("    avg round:  %8.3f ms  (from total / N)\n\n", total_ms / N_rounds);

    if (!cfg.overlap) {
        float ep_overhead = avg_scatter + avg_dispatch + avg_combine;
        std::printf("  EP overhead (scatter+dispatch+combine): %.3f ms (%.1f%% of round)\n",
                    ep_overhead, ep_overhead / avg_round * 100.f);
        std::printf("  FFN fraction: %.1f%%\n\n", avg_ffn / avg_round * 100.f);
    }

    // ── L2-norm checksum (round 0 only) — for cross-config correctness check
    double total_sum_sq = 0.0;
    long long total_elts = 0;
    for (int r = 0; r < n_ranks; ++r) {
        total_sum_sq += results[r].local_sum_sq;
        total_elts   += results[r].n_elts;
    }
    double l2 = std::sqrt(total_sum_sq);
    std::printf("  L2 checksum (round 0 d_combine_recv across all ranks):\n");
    std::printf("    n_elts = %lld   sum_sq = %.6e   L2 = %.6f\n\n",
                total_elts, total_sum_sq, l2);

    for (int r = 0; r < n_ranks; ++r)
        NCCL_CHECK(ncclCommDestroy(comms[r]));

    return 0;
}
