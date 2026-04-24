// apps/bench_ep.cu
//
// Multi-GPU Expert Parallelism benchmark.
//
// Measures the latency of each stage in a single MoE layer forward pass
// across N GPUs using Expert Parallelism:
//
//   Stage 1 — Local Scatter    : each GPU packs its T tokens by expert
//   Stage 2 — AllToAll Dispatch: tokens sent to the GPU owning each expert
//   Stage 3 — AllToAll Combine : expert outputs sent back to origin GPU
//
// FFN is not benchmarked here; its single-GPU time is already known from
// bench_e2e. The goal is to show how scatter + communication overhead
// compares to FFN time as GPU count increases.
//
// Usage:
//   ./build/bin/bench_ep --prefix=syn_uniform_T2048 --strategy=sort
//   ./build/bin/bench_ep --prefix=syn_zipf_T8192   --strategy=warp --n-gpus=4

#include "moe/cpu_reference.hpp"
#include "moe/npy_io.hpp"
#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <nccl.h>
#include <cuda_fp16.h>
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

// ── NCCL error check ─────────────────────────────────────────────────────────

#define NCCL_CHECK(call)                                                        \
    do {                                                                        \
        ncclResult_t _r = (call);                                               \
        if (_r != ncclSuccess) {                                                \
            std::fprintf(stderr, "NCCL error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, ncclGetErrorString(_r));           \
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
    BarrierN(int n) : n(n) {}

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

// ── Per-run result ────────────────────────────────────────────────────────────

struct EPResult {
    float scatter_ms  = 0.f;
    float dispatch_ms = 0.f;
    float combine_ms  = 0.f;
    float total_ms    = 0.f;
};

// ── CLI config ────────────────────────────────────────────────────────────────

struct Config {
    std::string prefix    = "";
    std::string data_dir  = "results";
    std::string strategy  = "sort";
    int         E_cli     = 64;
    int         warmup    = 5;
    int         iters     = 20;
    int         n_gpus    = 2;
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
        else if (k == "warmup")   c.warmup   = std::atoi(v.c_str());
        else if (k == "iters")    c.iters    = std::atoi(v.c_str());
        else if (k == "n-gpus")   c.n_gpus   = std::atoi(v.c_str());
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
    EPResult*     out)
{
    MOE_CUDA_CHECK(cudaSetDevice(rank));

    // ── Load routing data ─────────────────────────────────────────────────────
    auto emb  = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_embeddings.npy");
    auto asgn = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_assignments.npy");
    auto wts  = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_weights.npy");

    moe::MoEParams p{};
    p.T       = (int)emb.rows;
    p.K       = (int)asgn.cols;
    p.d_model = (int)emb.cols;
    p.d_ffn   = 0;

    const int32_t* h_asgn = asgn.as_i32();
    int32_t max_e = -1;
    for (std::size_t i = 0; i < asgn.rows * asgn.cols; ++i)
        max_e = std::max(max_e, h_asgn[i]);
    p.E = std::max(cfg.E_cli, (int)(max_e + 1));

    const int TK = p.T * p.K;
    const std::size_t chunk = (std::size_t)(TK / n_ranks) * p.d_model;

    // Convert weights to fp32
    std::vector<float> wts_f32(wts.rows * wts.cols);
    if (wts.dtype == "<f4") {
        std::memcpy(wts_f32.data(), wts.data.data(), wts_f32.size() * 4);
    } else {
        const __half* h = reinterpret_cast<const __half*>(wts.data.data());
        for (std::size_t i = 0; i < wts_f32.size(); ++i)
            wts_f32[i] = __half2float(h[i]);
    }

    // ── Device allocations ────────────────────────────────────────────────────
    __half*  d_emb  = moe::device_alloc<__half>((std::size_t)p.T * p.d_model);
    int32_t* d_asgn = moe::device_alloc<int32_t>((std::size_t)TK);
    float*   d_wts  = moe::device_alloc<float>((std::size_t)TK);

    MOE_CUDA_CHECK(cudaMemcpy(d_emb,  emb.as_fp16(),
        (std::size_t)p.T * p.d_model * 2, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_asgn, h_asgn,
        (std::size_t)TK * 4, cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_wts,  wts_f32.data(),
        (std::size_t)TK * 4, cudaMemcpyHostToDevice));

    moe::MoEBuffers bufs{};
    bufs.packed         = moe::device_alloc<__half>((std::size_t)TK * p.d_model);
    bufs.packed_weights = moe::device_alloc<float>((std::size_t)TK);
    bufs.perm           = moe::device_alloc<int32_t>((std::size_t)TK);
    bufs.expert_count   = moe::device_alloc<int32_t>((std::size_t)p.E);
    bufs.expert_start   = moe::device_alloc<int32_t>((std::size_t)(p.E + 1));
    bufs.strategy_ws_bytes = moe::make_scatter(cfg.strategy)->workspace_bytes(p);
    MOE_CUDA_CHECK(cudaMalloc(&bufs.strategy_ws, bufs.strategy_ws_bytes));

    __half* d_dispatch_recv = moe::device_alloc<__half>((std::size_t)TK * p.d_model);
    __half* d_combine_recv  = moe::device_alloc<__half>((std::size_t)TK * p.d_model);

    auto scatter = moe::make_scatter(cfg.strategy);

    cudaStream_t stream;
    MOE_CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Warmup ────────────────────────────────────────────────────────────────
    for (int i = 0; i < cfg.warmup; ++i) {
        scatter->run(d_emb, d_asgn, d_wts, bufs, p, stream);
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));

        barrier.wait();
        NCCL_CHECK(ncclGroupStart());
        for (int r = 0; r < n_ranks; ++r) {
            NCCL_CHECK(ncclSend(
                (const char*)bufs.packed + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
            NCCL_CHECK(ncclRecv(
                (char*)d_dispatch_recv + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));

        barrier.wait();
        NCCL_CHECK(ncclGroupStart());
        for (int r = 0; r < n_ranks; ++r) {
            NCCL_CHECK(ncclSend(
                (const char*)d_dispatch_recv + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
            NCCL_CHECK(ncclRecv(
                (char*)d_combine_recv + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));
        barrier.wait();
    }

    // ── Timed runs ────────────────────────────────────────────────────────────
    std::vector<float> v_scatter(cfg.iters), v_dispatch(cfg.iters),
                       v_combine(cfg.iters);

    for (int i = 0; i < cfg.iters; ++i) {
        moe::CudaTimer ts, td, tc;

        barrier.wait();
        ts.start(stream);
        scatter->run(d_emb, d_asgn, d_wts, bufs, p, stream);
        v_scatter[i] = ts.stop_ms(stream);

        barrier.wait();
        td.start(stream);
        NCCL_CHECK(ncclGroupStart());
        for (int r = 0; r < n_ranks; ++r) {
            NCCL_CHECK(ncclSend(
                (const char*)bufs.packed + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
            NCCL_CHECK(ncclRecv(
                (char*)d_dispatch_recv + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));
        v_dispatch[i] = td.stop_ms(stream);

        barrier.wait();
        tc.start(stream);
        NCCL_CHECK(ncclGroupStart());
        for (int r = 0; r < n_ranks; ++r) {
            NCCL_CHECK(ncclSend(
                (const char*)d_dispatch_recv + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
            NCCL_CHECK(ncclRecv(
                (char*)d_combine_recv + (std::size_t)r * chunk * sizeof(__half),
                chunk, ncclHalf, r, comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        MOE_CUDA_CHECK(cudaStreamSynchronize(stream));
        v_combine[i] = tc.stop_ms(stream);

        barrier.wait();
    }

    // ── Compute medians ───────────────────────────────────────────────────────
    auto median = [](std::vector<float>& v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    out->scatter_ms  = median(v_scatter);
    out->dispatch_ms = median(v_dispatch);
    out->combine_ms  = median(v_combine);
    out->total_ms    = out->scatter_ms + out->dispatch_ms + out->combine_ms;

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_emb); cudaFree(d_asgn); cudaFree(d_wts);
    cudaFree(bufs.packed); cudaFree(bufs.packed_weights); cudaFree(bufs.perm);
    cudaFree(bufs.expert_count); cudaFree(bufs.expert_start);
    cudaFree(bufs.strategy_ws);
    cudaFree(d_dispatch_recv); cudaFree(d_combine_recv);
    cudaStreamDestroy(stream);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Config cfg = parse(argc, argv);

    int n_gpus_avail = 0;
    MOE_CUDA_CHECK(cudaGetDeviceCount(&n_gpus_avail));
    if (n_gpus_avail < cfg.n_gpus) {
        std::fprintf(stderr, "Need %d GPUs, found %d\n", cfg.n_gpus, n_gpus_avail);
        return 1;
    }
    const int n_ranks = cfg.n_gpus;

    std::printf("\n== bench_ep  strategy=%s  prefix=%s  n_gpus=%d ==\n\n",
                cfg.strategy.c_str(), cfg.prefix.c_str(), n_ranks);

    // ── NCCL init ─────────────────────────────────────────────────────────────
    std::vector<ncclComm_t> comms(n_ranks);
    std::vector<int> dev_ids(n_ranks);
    for (int i = 0; i < n_ranks; i++) dev_ids[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), n_ranks, dev_ids.data()));

    // ── Launch per-rank threads ────────────────────────────────────────────────
    BarrierN barrier(n_ranks);
    std::vector<EPResult> results(n_ranks);
    std::vector<std::thread> threads;
    for (int r = 0; r < n_ranks; r++) {
        threads.emplace_back(rank_worker, r, n_ranks, comms[r],
                             std::ref(barrier), std::cref(cfg), &results[r]);
    }
    for (auto& t : threads) t.join();

    // ── Print results (average over all ranks) ────────────────────────────────
    float scatter_ms = 0, dispatch_ms = 0, combine_ms = 0;
    for (int r = 0; r < n_ranks; r++) {
        scatter_ms  += results[r].scatter_ms;
        dispatch_ms += results[r].dispatch_ms;
        combine_ms  += results[r].combine_ms;
    }
    scatter_ms  /= n_ranks;
    dispatch_ms /= n_ranks;
    combine_ms  /= n_ranks;
    float total_ms = scatter_ms + dispatch_ms + combine_ms;

    std::printf("  warmup=%d  iters=%d  (median over iters, avg over %d GPUs)\n\n",
                cfg.warmup, cfg.iters, n_ranks);
    std::printf("  scatter:   %8.3f ms  (%5.1f%%)\n",
                scatter_ms,  scatter_ms  / total_ms * 100.f);
    std::printf("  dispatch:  %8.3f ms  (%5.1f%%)\n",
                dispatch_ms, dispatch_ms / total_ms * 100.f);
    std::printf("  combine:   %8.3f ms  (%5.1f%%)\n",
                combine_ms,  combine_ms  / total_ms * 100.f);
    std::printf("  ─────────────────────────────────\n");
    std::printf("  total:     %8.3f ms\n\n", total_ms);

    std::printf("  Note: FFN time excluded (see bench_e2e for single-GPU FFN latency).\n");
    std::printf("  EP overhead = scatter + dispatch + combine\n");
    std::printf("  Compare against single-GPU scatter + FFN/N to see packing fraction growth.\n\n");

    for (int r = 0; r < n_ranks; ++r)
        NCCL_CHECK(ncclCommDestroy(comms[r]));

    return 0;
}
