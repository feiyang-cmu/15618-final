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
    int         warmup    = 2;
    int         repeats   = 5;
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

    cudaStream_t stream;
    MOE_CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    __half alpha_h = __float2half(1.f), beta_h = __float2half(0.f);

    // Host-side scratch (reused across rounds)
    std::vector<int32_t> h_all_counts((std::size_t)n_ranks * p.E);
    std::vector<int32_t> h_send_off_tok(n_ranks + 1);  // token offsets in bufs.packed
    std::vector<int32_t> h_recv_off_tok(n_ranks + 1);  // token offsets in d_dispatch_recv

    // ── Lambda: run one full round ────────────────────────────────────────────
    auto run_round = [&](int round_idx, RoundResult* rr) {
        int32_t* d_asgn = d_asgn_local + (std::size_t)round_idx * TK_local;
        float*   d_wts  = d_wts_local  + (std::size_t)round_idx * TK_local;

        moe::CudaTimer ts, td, tf, tc;

        // ── Stage 1: Local scatter ───────────────────────────────────────────
        barrier.wait();
        ts.start(stream);
        scatter->run(d_emb_local, d_asgn, d_wts, bufs, p, stream);
        rr->scatter_ms = ts.stop_ms(stream);

        // ── Stage 2: AllGather counts + AllToAll-V dispatch ──────────────────
        barrier.wait();
        td.start(stream);

        // 2a. AllGather every rank's expert_count
        NCCL_CHECK(ncclAllGather(
            bufs.expert_count, d_all_counts, p.E, ncclInt32, comm, stream));

        // 2b. Pull counts to host so we can compute per-pair send/recv sizes.
        //     This forces a stream sync between AllGather and ncclSend/Recv,
        //     which is unavoidable since NCCL needs host-known counts.
        MOE_CUDA_CHECK(cudaMemcpyAsync(
            h_all_counts.data(), d_all_counts,
            (std::size_t)n_ranks * p.E * 4, cudaMemcpyDeviceToHost, stream));
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));

        // 2c. Compute send offsets: this GPU's local packed tokens going to
        //     rank g are those for experts [g*epp, (g+1)*epp).
        h_send_off_tok[0] = 0;
        for (int g = 0; g < n_ranks; ++g) {
            int32_t s = 0;
            for (int e = g * experts_per_gpu; e < (g + 1) * experts_per_gpu; ++e)
                s += h_all_counts[(std::size_t)rank * p.E + e];
            h_send_off_tok[g + 1] = h_send_off_tok[g] + s;
        }

        // 2d. Compute recv offsets: from sender g, tokens for our local
        //     experts [rank*epp, (rank+1)*epp).
        h_recv_off_tok[0] = 0;
        for (int g = 0; g < n_ranks; ++g) {
            int32_t s = 0;
            for (int e = rank * experts_per_gpu; e < (rank + 1) * experts_per_gpu; ++e)
                s += h_all_counts[(std::size_t)g * p.E + e];
            h_recv_off_tok[g + 1] = h_recv_off_tok[g] + s;
        }

        // 2e. AllToAll-V via grouped ncclSend/ncclRecv with per-pair counts.
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < n_ranks; ++g) {
            std::size_t send_cnt =
                (std::size_t)(h_send_off_tok[g + 1] - h_send_off_tok[g]) * d_model;
            std::size_t recv_cnt =
                (std::size_t)(h_recv_off_tok[g + 1] - h_recv_off_tok[g]) * d_model;
            if (send_cnt > 0) {
                NCCL_CHECK(ncclSend(
                    bufs.packed + (std::size_t)h_send_off_tok[g] * d_model,
                    send_cnt, ncclHalf, g, comm, stream));
            }
            if (recv_cnt > 0) {
                NCCL_CHECK(ncclRecv(
                    d_dispatch_recv + (std::size_t)h_recv_off_tok[g] * d_model,
                    recv_cnt, ncclHalf, g, comm, stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));
        rr->dispatch_ms = td.stop_ms(stream);

        // ── Stage 3: FFN on dispatched data ─────────────────────────────────
        // Layout of d_dispatch_recv: chunks from each sender concatenated.
        // Within sender g's chunk, tokens are sorted by expert (g's local
        // scatter ordering), covering experts [rank*epp, (rank+1)*epp).
        // Run one cublasHgemm pair per (sender, local_expert).
        tf.start(stream);
        std::size_t curr_tok = 0;
        for (int g = 0; g < n_ranks; ++g) {
            for (int e_local = 0; e_local < experts_per_gpu; ++e_local) {
                int e_global = rank * experts_per_gpu + e_local;
                int cnt = h_all_counts[(std::size_t)g * p.E + e_global];
                if (cnt == 0) continue;

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
        rr->ffn_ms = tf.stop_ms(stream);

        // ── Stage 4: AllToAll-V combine (reverse of dispatch) ────────────────
        // Send d_expert_out chunks back to the GPU that originally sent them.
        barrier.wait();
        tc.start(stream);
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < n_ranks; ++g) {
            // Send back to g: the chunk we received from g, now FFN'd
            std::size_t send_cnt =
                (std::size_t)(h_recv_off_tok[g + 1] - h_recv_off_tok[g]) * d_model;
            // Receive from g: the chunk we originally sent to g, now FFN'd
            std::size_t recv_cnt =
                (std::size_t)(h_send_off_tok[g + 1] - h_send_off_tok[g]) * d_model;
            if (send_cnt > 0) {
                NCCL_CHECK(ncclSend(
                    d_expert_out + (std::size_t)h_recv_off_tok[g] * d_model,
                    send_cnt, ncclHalf, g, comm, stream));
            }
            if (recv_cnt > 0) {
                NCCL_CHECK(ncclRecv(
                    d_combine_recv + (std::size_t)h_send_off_tok[g] * d_model,
                    recv_cnt, ncclHalf, g, comm, stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));
        rr->combine_ms = tc.stop_ms(stream);

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
        t_total.start(stream);

        for (int r = 0; r < N_rounds; ++r) {
            run_round(r, &all_round_results[rep][r]);
        }

        total_times[rep] = t_total.stop_ms(stream);
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
    cudaStreamDestroy(stream);
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

    std::printf("\n== bench_prefill  strategy=%s  prefix=%s  n_gpus=%d ==\n\n",
                cfg.strategy.c_str(), cfg.prefix.c_str(), n_ranks);

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

    std::printf("  Full prefill (%d rounds):\n", N_rounds);
    std::printf("    total:     %8.3f ms\n", total_ms);
    std::printf("    throughput: %.1f rounds/s\n", N_rounds / (total_ms / 1000.f));
    std::printf("    avg round:  %8.3f ms  (from total / N)\n\n", total_ms / N_rounds);

    float ep_overhead = avg_scatter + avg_dispatch + avg_combine;
    std::printf("  EP overhead (scatter+dispatch+combine): %.3f ms (%.1f%% of round)\n",
                ep_overhead, ep_overhead / avg_round * 100.f);
    std::printf("  FFN fraction: %.1f%%\n\n", avg_ffn / avg_round * 100.f);

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
