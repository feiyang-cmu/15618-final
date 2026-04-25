// apps/bench_scatter.cu
//
// Multi-round benchmark harness for any IScatter strategy. Modeled on vLLM's
// benchmark_kernel.py and SGLang's microbench style:
//
//   1. allocate & upload inputs exactly once (fair to the strategy)
//   2. warmup W iterations to stabilize clocks & JIT lazy CUB kernels
//   3. timed loop on a dedicated CUDA stream, cudaEvent pair per iter
//   4. auto-scale iter count until total wall >= min_total_ms (default 50ms)
//      so microsecond kernels produce stable statistics
//   5. report min / median / p90 / p99 / mean / stdev
//
// Running on a non-default stream is deliberate — it's the same pattern we'll
// need for overlapping scatter with NCCL all-to-all in the multi-GPU stretch.
//
// Usage:
//   ./build/bin/bench_scatter --prefix=syn_zipf_T8192
//   ./build/bin/bench_scatter --prefix=syn_uniform_T2048 --strategy=atomic \
//                             --iters=200 --warmup=20

#include "moe/cpu_reference.hpp"
#include "moe/npy_io.hpp"
#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Config {
    std::string prefix;
    int         E_cli       = 8;
    std::string data_dir    = "results";
    std::string strategy    = "atomic";
    int         warmup      = 10;
    int         iters       = 0;        ///< 0 → auto-scale
    float       min_total_ms = 50.f;    ///< auto-scale target
    int         max_iters   = 5000;     ///< auto-scale cap
};

// Minimal --flag=value parser. Unknown args abort.
Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (a.rfind("--", 0) != 0 || eq == std::string::npos) {
            std::fprintf(stderr, "bad arg '%s' — expected --flag=value\n", argv[i]);
            std::exit(1);
        }
        std::string k = a.substr(2, eq - 2);
        std::string v = a.substr(eq + 1);
        if      (k == "prefix")       c.prefix   = v;
        else if (k == "E")            c.E_cli    = std::atoi(v.c_str());
        else if (k == "data-dir")     c.data_dir = v;
        else if (k == "strategy")     c.strategy = v;
        else if (k == "warmup")       c.warmup   = std::atoi(v.c_str());
        else if (k == "iters")        c.iters    = std::atoi(v.c_str());
        else if (k == "min-total-ms") c.min_total_ms = (float)std::atof(v.c_str());
        else if (k == "max-iters")    c.max_iters = std::atoi(v.c_str());
        else { std::fprintf(stderr, "unknown flag --%s\n", k.c_str()); std::exit(1); }
    }
    if (c.prefix.empty()) {
        std::fprintf(stderr, "usage: --prefix=<name> [--strategy=...] [--iters=N] [--warmup=W]\n");
        std::exit(1);
    }
    return c;
}

// ----------------------------------------------------------------------------
// Persistent device-side context: allocate once, reuse across iters.
// ----------------------------------------------------------------------------
struct DeviceCtx {
    __half*       d_emb         = nullptr;
    std::int32_t* d_asgn        = nullptr;
    float*        d_wts         = nullptr;
    moe::MoEBuffers bufs{};
    cudaStream_t  stream        = nullptr;

    void free() {
        cudaFree(d_emb); cudaFree(d_asgn); cudaFree(d_wts);
        cudaFree(bufs.packed); cudaFree(bufs.packed_weights); cudaFree(bufs.perm);
        cudaFree(bufs.expert_count); cudaFree(bufs.expert_start);
        cudaFree(bufs.strategy_ws);
        if (stream) cudaStreamDestroy(stream);
    }
};

DeviceCtx allocate_device(const moe::MoEParams& p, std::size_t ws_bytes) {
    DeviceCtx d;
    const int TK = p.T * p.K;
    d.d_emb  = moe::device_alloc<__half>((std::size_t)p.T * p.d_model);
    d.d_asgn = moe::device_alloc<std::int32_t>((std::size_t)TK);
    d.d_wts  = moe::device_alloc<float>((std::size_t)TK);
    d.bufs.packed         = moe::device_alloc<__half>((std::size_t)TK * p.d_model);
    d.bufs.packed_weights = moe::device_alloc<float>((std::size_t)TK);
    d.bufs.perm           = moe::device_alloc<std::int32_t>((std::size_t)TK);
    d.bufs.expert_count   = moe::device_alloc<std::int32_t>((std::size_t)p.E);
    d.bufs.expert_start   = moe::device_alloc<std::int32_t>((std::size_t)(p.E + 1));
    d.bufs.strategy_ws_bytes = ws_bytes;
    MOE_CUDA_CHECK(cudaMalloc(&d.bufs.strategy_ws, ws_bytes));
    MOE_CUDA_CHECK(cudaStreamCreate(&d.stream));
    return d;
}

// ----------------------------------------------------------------------------
// Statistics over an unsorted sample vector (modifies it).
// ----------------------------------------------------------------------------
struct Stats {
    float min, p50, p90, p99, mean, stdev;
};
Stats compute_stats(std::vector<float>& samples) {
    auto& s = samples;
    std::sort(s.begin(), s.end());
    const std::size_t n = s.size();
    auto at = [&](double q) { return s[std::min(n - 1, (std::size_t)(q * n)) ]; };
    double sum = std::accumulate(s.begin(), s.end(), 0.0);
    double mean = sum / n;
    double var = 0.0;
    for (float x : s) { double d = x - mean; var += d * d; }
    var /= n;
    return { s.front(), at(0.50), at(0.90), at(0.99), (float)mean, (float)std::sqrt(var) };
}

} // namespace

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg = parse(argc, argv);

    // 1. Load inputs.
    auto emb  = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_embeddings.npy");
    auto asgn = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_assignments.npy");
    auto wts  = moe::load_npy(cfg.data_dir + "/" + cfg.prefix + "_weights.npy");
    if (emb.dtype != "<f2" || asgn.dtype != "<i4" ||
        (wts.dtype != "<f4" && wts.dtype != "<f2")) {
        std::fprintf(stderr, "unexpected dtypes\n"); return 1;
    }

    moe::MoEParams p{};
    p.T       = (int)emb.rows;
    p.K       = (int)asgn.cols;
    p.d_model = (int)emb.cols;
    p.d_ffn   = 0;
    const std::int32_t* a = asgn.as_i32();
    std::int32_t max_e = -1;
    for (std::size_t i = 0; i < asgn.rows * asgn.cols; ++i) max_e = std::max(max_e, a[i]);
    p.E = std::max(cfg.E_cli, (int)(max_e + 1));

    std::vector<float> wts_f32(wts.rows * wts.cols);
    if (wts.dtype == "<f4") {
        std::memcpy(wts_f32.data(), wts.data.data(), wts_f32.size() * sizeof(float));
    } else {
        const __half* h = reinterpret_cast<const __half*>(wts.data.data());
        for (std::size_t i = 0; i < wts_f32.size(); ++i) wts_f32[i] = __half2float(h[i]);
    }

    // 2. Construct strategy, allocate device state.
    auto scatter = moe::make_scatter(cfg.strategy);
    if (!scatter) { std::fprintf(stderr, "unknown strategy '%s'\n", cfg.strategy.c_str()); return 1; }

    const int TK = p.T * p.K;
    DeviceCtx d = allocate_device(p, scatter->workspace_bytes(p));

    MOE_CUDA_CHECK(cudaMemcpy(d.d_emb,  emb.as_fp16(),
                              (std::size_t)p.T * p.d_model * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d.d_asgn, a, (std::size_t)TK * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d.d_wts,  wts_f32.data(), (std::size_t)TK * sizeof(float),
                              cudaMemcpyHostToDevice));

    std::printf("\n== bench_scatter  strategy=%s  prefix=%s ==\n",
                scatter->name(), cfg.prefix.c_str());
    std::printf("   T=%d  K=%d  E=%d  d=%d  (packed rows=%d)\n\n",
                p.T, p.K, p.E, p.d_model, TK);

    // 3. One-shot correctness check before burning CPU on benchmarking.
    {
        scatter->run(d.d_emb, d.d_asgn, d.d_wts, d.bufs, p, d.stream);
        MOE_CUDA_CHECK(cudaStreamSynchronize(d.stream));

        std::vector<std::uint16_t> h_packed((std::size_t)TK * p.d_model);
        std::vector<float>         h_pw((std::size_t)TK);
        std::vector<std::int32_t>  h_start((std::size_t)p.E), h_count((std::size_t)p.E), h_perm((std::size_t)TK);
        MOE_CUDA_CHECK(cudaMemcpy(h_packed.data(), d.bufs.packed,
                                  (std::size_t)TK * p.d_model * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        MOE_CUDA_CHECK(cudaMemcpy(h_pw.data(),    d.bufs.packed_weights,
                                  (std::size_t)TK * sizeof(float),             cudaMemcpyDeviceToHost));
        MOE_CUDA_CHECK(cudaMemcpy(h_start.data(), d.bufs.expert_start,
                                  (std::size_t)p.E * sizeof(std::int32_t),     cudaMemcpyDeviceToHost));
        MOE_CUDA_CHECK(cudaMemcpy(h_count.data(), d.bufs.expert_count,
                                  (std::size_t)p.E * sizeof(std::int32_t),     cudaMemcpyDeviceToHost));
        MOE_CUDA_CHECK(cudaMemcpy(h_perm.data(),  d.bufs.perm,
                                  (std::size_t)TK * sizeof(std::int32_t),      cudaMemcpyDeviceToHost));

        bool ok = moe::verify_scatter_outputs(
            emb.as_fp16(), a, wts_f32.data(),
            h_packed.data(), h_pw.data(),
            h_start.data(), h_count.data(), h_perm.data(),
            p.T, p.K, p.E, p.d_model);
        std::printf("   correctness: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { d.free(); return 1; }
    }

    // 4. Warmup.
    for (int i = 0; i < cfg.warmup; ++i) {
        scatter->run(d.d_emb, d.d_asgn, d.d_wts, d.bufs, p, d.stream);
    }
    MOE_CUDA_CHECK(cudaStreamSynchronize(d.stream));

    // 5. Timed loop. Per-iter cudaEvent pair, pre-allocated.
    //    If cfg.iters == 0 we auto-scale until total >= cfg.min_total_ms.
    //    Also accumulates per-stage breakdown via ScatterTimings (only the
    //    stages the strategy reports — others stay 0).
    struct StageAcc {
        double count = 0, prefix = 0, sort = 0, gather = 0, total = 0;
    };
    auto time_iters = [&](int N, StageAcc* stages) {
        std::vector<cudaEvent_t> starts(N), stops(N);
        std::vector<moe::ScatterTimings> per_iter(N);
        for (int i = 0; i < N; ++i) {
            MOE_CUDA_CHECK(cudaEventCreate(&starts[i]));
            MOE_CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        for (int i = 0; i < N; ++i) {
            MOE_CUDA_CHECK(cudaEventRecord(starts[i], d.stream));
            per_iter[i] = scatter->run(d.d_emb, d.d_asgn, d.d_wts, d.bufs, p, d.stream);
            MOE_CUDA_CHECK(cudaEventRecord(stops[i], d.stream));
        }
        MOE_CUDA_CHECK(cudaStreamSynchronize(d.stream));

        std::vector<float> ms(N);
        for (int i = 0; i < N; ++i) {
            MOE_CUDA_CHECK(cudaEventElapsedTime(&ms[i], starts[i], stops[i]));
        }
        if (stages) {
            *stages = StageAcc{};
            for (const auto& t : per_iter) {
                stages->count  += t.count_ms;
                stages->prefix += t.prefix_ms;
                stages->sort   += t.sort_ms;
                stages->gather += t.gather_ms;
                stages->total  += t.total_ms;
            }
        }
        for (int i = 0; i < N; ++i) {
            cudaEventDestroy(starts[i]); cudaEventDestroy(stops[i]);
        }
        return ms;
    };

    int N = cfg.iters > 0 ? cfg.iters : 50;  // start point for auto-scale
    std::vector<float> samples;
    StageAcc stages{};
    while (true) {
        samples = time_iters(N, &stages);
        float total = std::accumulate(samples.begin(), samples.end(), 0.f);
        if (cfg.iters > 0) break;          // fixed count requested
        if (total >= cfg.min_total_ms) break;
        if (N >= cfg.max_iters) break;
        N = std::min(cfg.max_iters, N * 2);
    }

    // 6. Report.
    Stats st = compute_stats(samples);
    float total = st.mean * samples.size();
    std::printf("   iters=%zu  warmup=%d  total=%.2fms\n", samples.size(), cfg.warmup, total);
    std::printf("   time (ms) — min %.4f  p50 %.4f  p90 %.4f  p99 %.4f  mean %.4f  stdev %.4f\n",
                st.min, st.p50, st.p90, st.p99, st.mean, st.stdev);

    // Per-stage breakdown (only stages the strategy reports are non-zero).
    int M = (int)samples.size();
    double mc = stages.count / M, ms_sort = stages.sort / M, mp = stages.prefix / M,
           mg = stages.gather / M, mt = stages.total / M;
    if (mt > 0.0) {
        std::printf("   per-stage (mean ms)  count=%.4f  sort=%.4f  prefix=%.4f  gather=%.4f  | sum=%.4f  reported_total=%.4f\n",
                    mc, ms_sort, mp, mg, mc + ms_sort + mp + mg, mt);
    }

    // Effective bandwidth based on packed-row data movement (read emb, write packed).
    double bytes = 2.0 * (double)TK * p.d_model * sizeof(std::uint16_t);
    double gbps  = (bytes / 1e9) / (st.p50 / 1e3);
    std::printf("   effective bandwidth @ p50: %.1f GB/s  (%.2f MB moved)\n",
                gbps, bytes / (1024.0 * 1024.0));

    // Expert skew (informational; explains tail latency under zipf).
    std::vector<std::int32_t> h_count((std::size_t)p.E);
    MOE_CUDA_CHECK(cudaMemcpy(h_count.data(), d.bufs.expert_count,
                              (std::size_t)p.E * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    int min_c = INT32_MAX, max_c = 0;
    for (int e = 0; e < p.E; ++e) { min_c = std::min(min_c, h_count[e]); max_c = std::max(max_c, h_count[e]); }
    double mean_c = (double)TK / p.E;
    std::printf("   expert load: min=%d max=%d mean=%.1f imbalance=%.2fx\n\n",
                min_c, max_c, mean_c, max_c / mean_c);

    d.free();
    return 0;
}
