// apps/verify.cu
//
// Correctness harness for any IScatter strategy. Runs the same inputs through:
//   1. the CPU reference (moe::cpu_scatter) — checked via verify_scatter_outputs
//   2. the selected GPU strategy — checked via verify_scatter_outputs
//
// Both pass iff the output satisfies all scatter invariants. We do NOT compare
// packed rows element-for-element because atomic/warp strategies produce
// non-deterministic intra-expert orderings.
//
// Usage:
//   ./build/bin/verify                               # mini-tests only
//   ./build/bin/verify <prefix>                      # also run on results/<prefix>_*.npy
//   ./build/bin/verify <prefix> <E> <data_dir> <strategy>
//
// Defaults: E=8, data_dir=results, strategy=atomic.

#include "moe/cpu_reference.hpp"
#include "moe/npy_io.hpp"
#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Host-side helper: run the selected IScatter strategy and copy results back.
// ----------------------------------------------------------------------------
struct GpuResult {
    std::vector<std::uint16_t> packed;
    std::vector<float>         packed_weights;
    std::vector<std::int32_t>  expert_start;
    std::vector<std::int32_t>  expert_count;
    std::vector<std::int32_t>  perm;
    moe::ScatterTimings        timings;
};

static GpuResult run_gpu(moe::IScatter& scatter,
                         const std::uint16_t* h_emb,
                         const std::int32_t*  h_asgn,
                         const float*         h_wts,
                         const moe::MoEParams& p) {
    const int TK = p.T * p.K;

    // Upload inputs.
    __half*  d_emb  = moe::device_alloc<__half>((std::size_t)p.T * p.d_model);
    std::int32_t* d_asgn = moe::device_alloc<std::int32_t>((std::size_t)TK);
    float*   d_wts  = moe::device_alloc<float>((std::size_t)TK);
    MOE_CUDA_CHECK(cudaMemcpy(d_emb,  h_emb,
                              (std::size_t)p.T * p.d_model * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_asgn, h_asgn, (std::size_t)TK * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice));
    MOE_CUDA_CHECK(cudaMemcpy(d_wts,  h_wts,  (std::size_t)TK * sizeof(float),
                              cudaMemcpyHostToDevice));

    // Allocate outputs.
    moe::MoEBuffers b{};
    b.packed         = moe::device_alloc<__half>((std::size_t)TK * p.d_model);
    b.packed_weights = moe::device_alloc<float>((std::size_t)TK);
    b.perm           = moe::device_alloc<std::int32_t>((std::size_t)TK);
    b.expert_count   = moe::device_alloc<std::int32_t>((std::size_t)p.E);
    b.expert_start   = moe::device_alloc<std::int32_t>((std::size_t)(p.E + 1));
    b.strategy_ws_bytes = scatter.workspace_bytes(p);
    MOE_CUDA_CHECK(cudaMalloc(&b.strategy_ws, b.strategy_ws_bytes));

    auto timings = scatter.run(d_emb, d_asgn, d_wts, b, p, /*stream=*/0);

    GpuResult r;
    r.packed.resize((std::size_t)TK * p.d_model);
    r.packed_weights.resize((std::size_t)TK);
    r.expert_start.resize((std::size_t)p.E);
    r.expert_count.resize((std::size_t)p.E);
    r.perm.resize((std::size_t)TK);
    r.timings = timings;

    MOE_CUDA_CHECK(cudaMemcpy(r.packed.data(), b.packed,
                              (std::size_t)TK * p.d_model * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
    MOE_CUDA_CHECK(cudaMemcpy(r.packed_weights.data(), b.packed_weights,
                              (std::size_t)TK * sizeof(float),
                              cudaMemcpyDeviceToHost));
    MOE_CUDA_CHECK(cudaMemcpy(r.expert_start.data(), b.expert_start,
                              (std::size_t)p.E * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
    MOE_CUDA_CHECK(cudaMemcpy(r.expert_count.data(), b.expert_count,
                              (std::size_t)p.E * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
    MOE_CUDA_CHECK(cudaMemcpy(r.perm.data(), b.perm,
                              (std::size_t)TK * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));

    cudaFree(b.packed); cudaFree(b.packed_weights); cudaFree(b.perm);
    cudaFree(b.expert_count); cudaFree(b.expert_start); cudaFree(b.strategy_ws);
    cudaFree(d_emb); cudaFree(d_asgn); cudaFree(d_wts);
    return r;
}

// ----------------------------------------------------------------------------
// Mini-tests: tiny synthetic inputs, check both CPU ref and GPU satisfy
// invariants.
// ----------------------------------------------------------------------------
static bool run_mini_tests(const std::string& strategy_name) {
    std::printf("\n=== Mini-tests (T=16, E=4, K=2, d=8, strategy=%s) ===\n\n",
                strategy_name.c_str());

    moe::MoEParams p{/*T=*/16, /*K=*/2, /*E=*/4, /*d_model=*/8, /*d_ffn=*/0};
    const int TK = p.T * p.K;

    std::vector<std::uint16_t> h_emb((std::size_t)p.T * p.d_model);
    std::vector<std::int32_t>  h_asgn((std::size_t)TK);
    std::vector<float>         h_wts((std::size_t)TK);
    for (int t = 0; t < p.T; ++t)
        for (int i = 0; i < p.d_model; ++i)
            h_emb[t * p.d_model + i] = (std::uint16_t)(t * 100 + i);
    for (int t = 0; t < p.T; ++t) {
        h_asgn[t * p.K + 0] = t % p.E;
        h_asgn[t * p.K + 1] = (t + 1) % p.E;
        h_wts[t * p.K + 0]  = 0.3f + 0.001f * t;
        h_wts[t * p.K + 1]  = 0.7f - 0.001f * t;
    }

    // CPU reference.
    std::vector<std::uint16_t> cpu_packed((std::size_t)TK * p.d_model);
    std::vector<float>         cpu_pw((std::size_t)TK);
    std::vector<std::int32_t>  cpu_start((std::size_t)p.E), cpu_count((std::size_t)p.E), cpu_perm((std::size_t)TK);
    moe::cpu_scatter(h_emb.data(), h_asgn.data(), h_wts.data(),
                     p.T, p.K, p.E, p.d_model,
                     cpu_packed.data(), cpu_pw.data(),
                     cpu_start.data(), cpu_count.data(), cpu_perm.data());

    bool cpu_ok = moe::verify_scatter_outputs(
        h_emb.data(), h_asgn.data(), h_wts.data(),
        cpu_packed.data(), cpu_pw.data(),
        cpu_start.data(), cpu_count.data(), cpu_perm.data(),
        p.T, p.K, p.E, p.d_model);
    std::printf("  cpu_scatter:      %s\n", cpu_ok ? "PASS" : "FAIL");
    if (!cpu_ok) return false;

    // GPU strategy.
    auto scatter = moe::make_scatter(strategy_name);
    if (!scatter) { std::printf("  unknown strategy '%s'\n", strategy_name.c_str()); return false; }
    GpuResult g = run_gpu(*scatter, h_emb.data(), h_asgn.data(), h_wts.data(), p);
    bool gpu_ok = moe::verify_scatter_outputs(
        h_emb.data(), h_asgn.data(), h_wts.data(),
        g.packed.data(), g.packed_weights.data(),
        g.expert_start.data(), g.expert_count.data(), g.perm.data(),
        p.T, p.K, p.E, p.d_model);
    std::printf("  %-18s%s  (count=%.3fms prefix=%.3fms gather=%.3fms)\n",
                (std::string(scatter->name()) + ":").c_str(),
                gpu_ok ? "PASS" : "FAIL",
                g.timings.count_ms, g.timings.prefix_ms, g.timings.gather_ms);
    return gpu_ok;
}

// ----------------------------------------------------------------------------
// Full run: load .npy inputs (embeddings, assignments, weights).
// ----------------------------------------------------------------------------
static bool run_full(const std::string& prefix,
                     int E_cli,
                     const std::string& data_dir,
                     const std::string& strategy_name) {
    std::printf("\n=== Full run: %s  (strategy=%s) ===\n\n",
                prefix.c_str(), strategy_name.c_str());

    auto emb  = moe::load_npy(data_dir + "/" + prefix + "_embeddings.npy");
    auto asgn = moe::load_npy(data_dir + "/" + prefix + "_assignments.npy");
    auto wts  = moe::load_npy(data_dir + "/" + prefix + "_weights.npy");

    if (emb.dtype != "<f2" || asgn.dtype != "<i4" ||
        (wts.dtype != "<f4" && wts.dtype != "<f2")) {
        std::fprintf(stderr, "unexpected dtypes: emb=%s asgn=%s wts=%s\n",
                     emb.dtype.c_str(), asgn.dtype.c_str(), wts.dtype.c_str());
        return false;
    }
    if (emb.rows != asgn.rows || asgn.rows != wts.rows || asgn.cols != wts.cols) {
        std::fprintf(stderr, "shape mismatch across npy files\n");
        return false;
    }

    // Normalize weights to fp32 for the downstream pipeline. The generator
    // stores them as fp16 to match the Qwen routing format.
    std::vector<float> wts_f32(wts.rows * wts.cols);
    if (wts.dtype == "<f4") {
        std::memcpy(wts_f32.data(), wts.data.data(), wts_f32.size() * sizeof(float));
    } else {
        const __half* h = reinterpret_cast<const __half*>(wts.data.data());
        for (std::size_t i = 0; i < wts_f32.size(); ++i) wts_f32[i] = __half2float(h[i]);
    }
    const float* wts_ptr = wts_f32.data();

    moe::MoEParams p{};
    p.T       = (int)emb.rows;
    p.K       = (int)asgn.cols;
    p.d_model = (int)emb.cols;
    p.d_ffn   = 0;
    const std::int32_t* a = asgn.as_i32();
    std::int32_t max_e = -1;
    for (std::size_t i = 0; i < asgn.rows * asgn.cols; ++i)
        if (a[i] > max_e) max_e = a[i];
    p.E = std::max(E_cli, (int)(max_e + 1));

    std::printf("\nDims: T=%d  K=%d  E=%d  d=%d  (packed rows=%d)\n\n",
                p.T, p.K, p.E, p.d_model, p.T * p.K);

    const int TK = p.T * p.K;

    // CPU reference (acts as a sanity check on the loader + invariants).
    std::vector<std::uint16_t> cpu_packed((std::size_t)TK * p.d_model);
    std::vector<float>         cpu_pw((std::size_t)TK);
    std::vector<std::int32_t>  cpu_start((std::size_t)p.E), cpu_count((std::size_t)p.E), cpu_perm((std::size_t)TK);
    moe::cpu_scatter(emb.as_fp16(), a, wts_ptr,
                     p.T, p.K, p.E, p.d_model,
                     cpu_packed.data(), cpu_pw.data(),
                     cpu_start.data(), cpu_count.data(), cpu_perm.data());
    bool cpu_ok = moe::verify_scatter_outputs(
        emb.as_fp16(), a, wts_ptr,
        cpu_packed.data(), cpu_pw.data(),
        cpu_start.data(), cpu_count.data(), cpu_perm.data(),
        p.T, p.K, p.E, p.d_model);
    std::printf("cpu_scatter invariants: %s\n", cpu_ok ? "PASS" : "FAIL");
    if (!cpu_ok) return false;

    // GPU strategy.
    auto scatter = moe::make_scatter(strategy_name);
    if (!scatter) { std::fprintf(stderr, "unknown strategy '%s'\n", strategy_name.c_str()); return false; }

    // One warmup, then one timed run.
    GpuResult g = run_gpu(*scatter, emb.as_fp16(), a, wts_ptr, p);
    g           = run_gpu(*scatter, emb.as_fp16(), a, wts_ptr, p);

    bool gpu_ok = moe::verify_scatter_outputs(
        emb.as_fp16(), a, wts_ptr,
        g.packed.data(), g.packed_weights.data(),
        g.expert_start.data(), g.expert_count.data(), g.perm.data(),
        p.T, p.K, p.E, p.d_model);

    std::printf("\nTiming (%s): count=%.3fms prefix=%.3fms gather=%.3fms  total=%.3fms\n",
                scatter->name(),
                g.timings.count_ms, g.timings.prefix_ms,
                g.timings.gather_ms, g.timings.total_ms);

    // Expert load summary (useful for skew sanity).
    int min_c = INT32_MAX, max_c = 0; long long sum_c = 0;
    for (int e = 0; e < p.E; ++e) {
        min_c = std::min(min_c, g.expert_count[e]);
        max_c = std::max(max_c, g.expert_count[e]);
        sum_c += g.expert_count[e];
    }
    double mean_c = (double)sum_c / p.E;
    std::printf("Expert load: min=%d  max=%d  mean=%.1f  imbalance=%.2fx\n",
                min_c, max_c, mean_c, mean_c > 0 ? max_c / mean_c : 0.0);

    std::printf("\n%s invariants: %s\n", scatter->name(), gpu_ok ? "PASS" : "FAIL");
    return gpu_ok;
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string prefix    = (argc > 1) ? argv[1] : "";
    int         E_cli     = (argc > 2) ? std::atoi(argv[2]) : 8;
    std::string data_dir  = (argc > 3) ? argv[3] : "results";
    std::string strategy  = (argc > 4) ? argv[4] : "atomic";

    if (!run_mini_tests(strategy)) return 1;
    if (!prefix.empty()) {
        if (!run_full(prefix, E_cli, data_dir, strategy)) return 1;
    }
    std::printf("\nALL PASS\n");
    return 0;
}
