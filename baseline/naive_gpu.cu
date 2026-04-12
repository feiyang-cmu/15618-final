// baseline/naive_gpu.cu
//
// Naive GPU baseline: direct port of the sequential scatter to CUDA.
// Three separate kernels — count, prefix sum (CUB), scatter — no fusion.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 baseline/naive_gpu.cu -o baseline/naive_gpu -lnvToolsExt
// Run:
//   ./baseline/naive_gpu                        # mini-tests only
//   ./baseline/naive_gpu syn_uniform_T2048      # run on real data
//   ./baseline/naive_gpu syn_zipf_T8192 64 results

#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <nvToolsExt.h>

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------

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
// .npy loader (same as sequential_cpu.cpp, duplicated to keep files standalone)
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
    if (std::fread(magic, 1, 6, fp) != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "[npy] bad magic in %s\n", path.c_str()); std::exit(1);
    }

    uint8_t major = 0, minor = 0;
    if (std::fread(&major, 1, 1, fp) != 1) { std::exit(1); }
    if (std::fread(&minor, 1, 1, fp) != 1) { std::exit(1); }

    size_t header_len = 0;
    if (major == 1) { uint16_t h; if (std::fread(&h, 2, 1, fp) != 1) std::exit(1); header_len = h; }
    else            { uint32_t h; if (std::fread(&h, 4, 1, fp) != 1) std::exit(1); header_len = h; }

    std::string header(header_len, ' ');
    if (std::fread(&header[0], 1, header_len, fp) != header_len) std::exit(1);

    NpyArray arr;
    auto d_pos = header.find("'descr':"); auto q1 = header.find('\'', d_pos+8); auto q2 = header.find('\'', q1+1);
    arr.dtype = header.substr(q1+1, q2-q1-1);

    auto s_pos = header.find("'shape':"); auto lp = header.find('(', s_pos); auto rp = header.find(')', lp);
    std::string shape_str = header.substr(lp+1, rp-lp-1);
    size_t comma = shape_str.find(',');
    arr.rows = std::strtoul(shape_str.substr(0, comma).c_str(), nullptr, 10);
    arr.cols = std::strtoul(shape_str.substr(comma+1).c_str(), nullptr, 10);

    size_t elt = (arr.dtype == "<f2") ? 2 : (arr.dtype == "<i4") ? 4 : 0;
    if (!elt) { std::fprintf(stderr, "[npy] unsupported dtype '%s'\n", arr.dtype.c_str()); std::exit(1); }

    size_t nbytes = arr.rows * arr.cols * elt;
    arr.data.resize(nbytes);
    if (std::fread(arr.data.data(), 1, nbytes, fp) != nbytes) { std::exit(1); }
    std::fclose(fp);
    std::printf("[npy] %-50s  (%zu, %zu)  %s\n", path.c_str(), arr.rows, arr.cols, arr.dtype.c_str());
    return arr;
}

// ---------------------------------------------------------------------------
// CPU reference (for verification in mini-tests and full runs)
// ---------------------------------------------------------------------------

static void cpu_scatter(
    const uint16_t* emb, const int32_t* asgn,
    int T, int K, int E, int d,
    uint16_t* packed, int32_t* expert_start, int32_t* expert_count, int32_t* perm)
{
    std::memset(expert_count, 0, E * sizeof(int32_t));
    for (int t = 0; t < T; t++)
        for (int k = 0; k < K; k++)
            expert_count[asgn[t*K + k]]++;

    int32_t run = 0;
    for (int e = 0; e < E; e++) { expert_start[e] = run; run += expert_count[e]; }

    std::vector<int32_t> cursor(E, 0);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            int e = asgn[t*K + k];
            int pos = expert_start[e] + cursor[e]++;
            std::memcpy(packed + (size_t)pos * d, emb + (size_t)t * d, d * sizeof(uint16_t));
            perm[pos] = t;
        }
    }
}

// ===========================================================================
// Kernel 1: count — one thread per (t,k), atomicAdd into expert_count
// ===========================================================================

__global__ void count_kernel(
    const int32_t* __restrict__ assignments,  // [T*K]
    int32_t*       __restrict__ expert_count, // [E], zeroed
    int TK)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TK) return;
    atomicAdd(&expert_count[assignments[idx]], 1);
}

// ===========================================================================
// Kernel 2: prefix sum — we just call CUB, no custom kernel needed
// ===========================================================================
// (see gpu_scatter below)

// ===========================================================================
// Kernel 3: scatter — one thread per (t,k), atomicAdd on cursor for write slot
// ===========================================================================
// Each thread:
//   1. atomicAdd on cursor[e] to grab a unique write position
//   2. copy d uint16 values from embeddings[t] to packed[pos]
//   3. write permutation[pos] = t
//
// The cursor is separate from expert_count because CUB already consumed
// expert_count to produce expert_start. We need fresh zeroed counters.

__global__ void scatter_kernel(
    const uint16_t* __restrict__ embeddings,    // [T, d]
    const int32_t*  __restrict__ assignments,   // [T, K]
    const int32_t*  __restrict__ expert_start,  // [E]
    int32_t*        __restrict__ cursor,         // [E], zeroed — per-expert write head
    uint16_t*       __restrict__ packed,         // [T*K, d]
    int32_t*        __restrict__ permutation,    // [T*K]
    int T, int K, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int TK = T * K;
    if (idx >= TK) return;

    int t = idx / K;
    int e = assignments[idx];

    // claim a write slot within this expert's region
    int slot = atomicAdd(&cursor[e], 1);
    int pos = expert_start[e] + slot;

    // copy the embedding row — not coalesced across threads, we know, that's
    // why this is the "naive" baseline
    const uint16_t* src = embeddings + (size_t)t * d;
    uint16_t*       dst = packed     + (size_t)pos * d;
    for (int i = 0; i < d; i++)
        dst[i] = src[i];

    permutation[pos] = t;
}

// ===========================================================================
// gpu_scatter: orchestrates the three steps
// ===========================================================================

struct GpuScatterResult {
    float count_ms;
    float prefix_ms;
    float scatter_ms;
};

static GpuScatterResult gpu_scatter(
    const uint16_t* d_emb,      // device [T*d]
    const int32_t*  d_asgn,     // device [T*K]
    int T, int K, int E, int d,
    uint16_t* d_packed,         // device [T*K*d], output
    int32_t*  d_expert_start,   // device [E], output
    int32_t*  d_expert_count,   // device [E], output
    int32_t*  d_permutation,    // device [T*K], output
    cudaStream_t stream = 0)
{
    int TK = T * K;
    int block = 256;
    int grid_tk = (TK + block - 1) / block;

    // temp cursor array for scatter (zeroed before use)
    int32_t* d_cursor = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cursor, E * sizeof(int32_t)));

    // CUB prefix sum needs temp storage — query size first
    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_cub_tmp, cub_bytes,
                                  d_expert_count, d_expert_start, E, stream);
    CHECK_CUDA(cudaMalloc(&d_cub_tmp, cub_bytes));

    // timing events
    cudaEvent_t t0, t1, t2, t3;
    CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventCreate(&t2)); CHECK_CUDA(cudaEventCreate(&t3));

    // --- step 1: count ---
    CHECK_CUDA(cudaMemsetAsync(d_expert_count, 0, E * sizeof(int32_t), stream));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    nvtxRangePush("count_kernel");
    count_kernel<<<grid_tk, block, 0, stream>>>(d_asgn, d_expert_count, TK);
    nvtxRangePop();
    CHECK_CUDA(cudaEventRecord(t1, stream));

    // --- step 2: exclusive prefix sum via CUB ---
    nvtxRangePush("cub_prefix_sum");
    cub::DeviceScan::ExclusiveSum(d_cub_tmp, cub_bytes,
                                  d_expert_count, d_expert_start, E, stream);
    nvtxRangePop();
    CHECK_CUDA(cudaEventRecord(t2, stream));

    // --- step 3: scatter ---
    CHECK_CUDA(cudaMemsetAsync(d_cursor, 0, E * sizeof(int32_t), stream));
    nvtxRangePush("scatter_kernel");
    scatter_kernel<<<grid_tk, block, 0, stream>>>(
        d_emb, d_asgn, d_expert_start, d_cursor,
        d_packed, d_permutation, T, K, d);
    nvtxRangePop();
    CHECK_CUDA(cudaEventRecord(t3, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    GpuScatterResult res;
    CHECK_CUDA(cudaEventElapsedTime(&res.count_ms,   t0, t1));
    CHECK_CUDA(cudaEventElapsedTime(&res.prefix_ms,  t1, t2));
    CHECK_CUDA(cudaEventElapsedTime(&res.scatter_ms, t2, t3));

    CHECK_CUDA(cudaFree(d_cursor));
    CHECK_CUDA(cudaFree(d_cub_tmp));
    CHECK_CUDA(cudaEventDestroy(t0)); CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaEventDestroy(t2)); CHECK_CUDA(cudaEventDestroy(t3));
    return res;
}

// ===========================================================================
// Verification: compare GPU output against CPU reference
// ===========================================================================
// We can't compare packed row-for-row because the GPU scatter uses atomics
// which give a non-deterministic ordering within each expert region. Instead:
//   1. expert_count must match exactly
//   2. expert_start must match exactly (both come from the same prefix sum)
//   3. for each expert region, the SET of (token_index, embedding_row) pairs
//      must match — order doesn't matter

static bool verify_gpu_vs_cpu(
    const uint16_t* h_emb,
    const int32_t*  h_asgn,
    const uint16_t* gpu_packed,
    const int32_t*  gpu_expert_start,
    const int32_t*  gpu_expert_count,
    const int32_t*  gpu_perm,
    const uint16_t* cpu_packed,
    const int32_t*  cpu_expert_start,
    const int32_t*  cpu_expert_count,
    const int32_t*  cpu_perm,
    int T, int K, int E, int d)
{
    int errors = 0;

    // expert_count must match
    for (int e = 0; e < E; e++) {
        if (gpu_expert_count[e] != cpu_expert_count[e]) {
            std::printf("  [err] expert_count[%d]: gpu=%d cpu=%d\n",
                        e, gpu_expert_count[e], cpu_expert_count[e]);
            if (++errors > 10) return false;
        }
    }

    // expert_start must match
    for (int e = 0; e < E; e++) {
        if (gpu_expert_start[e] != cpu_expert_start[e]) {
            std::printf("  [err] expert_start[%d]: gpu=%d cpu=%d\n",
                        e, gpu_expert_start[e], cpu_expert_start[e]);
            if (++errors > 10) return false;
        }
    }

    // per expert region: check that the set of token ids match and each
    // packed row matches its source embedding
    for (int e = 0; e < E; e++) {
        int start = gpu_expert_start[e];
        int count = gpu_expert_count[e];

        // collect token ids in GPU's region for this expert
        std::vector<int> gpu_tokens(count), cpu_tokens(count);
        for (int i = 0; i < count; i++) {
            gpu_tokens[i] = gpu_perm[start + i];
            cpu_tokens[i] = cpu_perm[start + i];
        }
        std::sort(gpu_tokens.begin(), gpu_tokens.end());
        std::sort(cpu_tokens.begin(), cpu_tokens.end());

        if (gpu_tokens != cpu_tokens) {
            std::printf("  [err] expert %d: token sets differ\n", e);
            if (++errors > 10) return false;
        }

        // check each GPU packed row matches the embedding of its claimed token
        for (int i = 0; i < count; i++) {
            int pos = start + i;
            int t = gpu_perm[pos];
            if (std::memcmp(gpu_packed + (size_t)pos * d,
                            h_emb + (size_t)t * d,
                            d * sizeof(uint16_t)) != 0) {
                std::printf("  [err] packed[%d] doesn't match embedding[%d]\n", pos, t);
                if (++errors > 10) return false;
            }
        }
    }

    return errors == 0;
}

// ===========================================================================
// Mini-tests: tiny inputs, check each kernel step against CPU
// ===========================================================================

static bool run_mini_tests() {
    std::printf("\n=== Mini-tests (T=16, E=4, K=2, d=8) ===\n\n");

    const int T = 16, E = 4, K = 2, d = 8;
    const int TK = T * K;

    // build deterministic test data on host
    std::vector<uint16_t> h_emb(T * d);
    std::vector<int32_t>  h_asgn(T * K);

    // embeddings: token t gets values [t*100 + 0, t*100 + 1, ... t*100 + d-1]
    // so we can easily identify which token a row came from
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            h_emb[t * d + i] = (uint16_t)(t * 100 + i);

    // assignments: token t -> experts (t % E) and ((t+1) % E)
    for (int t = 0; t < T; t++) {
        h_asgn[t * K + 0] = t % E;
        h_asgn[t * K + 1] = (t + 1) % E;
    }

    // CPU reference
    std::vector<uint16_t> cpu_packed(TK * d);
    std::vector<int32_t>  cpu_start(E), cpu_count(E), cpu_perm(TK);
    cpu_scatter(h_emb.data(), h_asgn.data(), T, K, E, d,
                cpu_packed.data(), cpu_start.data(), cpu_count.data(), cpu_perm.data());

    // allocate device memory
    uint16_t *d_emb, *d_packed;
    int32_t  *d_asgn, *d_expert_start, *d_expert_count, *d_perm;
    CHECK_CUDA(cudaMalloc(&d_emb,          T * d * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_asgn,         TK * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_packed,       TK * d * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_expert_start, E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_expert_count, E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_perm,         TK * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_emb,  h_emb.data(),  T * d * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_asgn, h_asgn.data(), TK * sizeof(int32_t),    cudaMemcpyHostToDevice));

    // --- test 1: count kernel ---
    {
        CHECK_CUDA(cudaMemset(d_expert_count, 0, E * sizeof(int32_t)));
        int block = 32, grid = (TK + block - 1) / block;
        count_kernel<<<grid, block>>>(d_asgn, d_expert_count, TK);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int32_t> gpu_count(E);
        CHECK_CUDA(cudaMemcpy(gpu_count.data(), d_expert_count, E * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));

        bool ok = (gpu_count == cpu_count);
        std::printf("  count_kernel:     %s\n", ok ? "PASS" : "FAIL");
        if (!ok) {
            for (int e = 0; e < E; e++)
                std::printf("    expert %d: gpu=%d cpu=%d\n", e, gpu_count[e], cpu_count[e]);
            return false;
        }
    }

    // --- test 2: CUB prefix sum ---
    {
        // expert_count already on device from test 1
        void*  d_tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_expert_count, d_expert_start, E);
        CHECK_CUDA(cudaMalloc(&d_tmp, tmp_bytes));
        cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_expert_count, d_expert_start, E);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int32_t> gpu_start(E);
        CHECK_CUDA(cudaMemcpy(gpu_start.data(), d_expert_start, E * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_tmp));

        bool ok = (gpu_start == cpu_start);
        std::printf("  cub_prefix_sum:   %s\n", ok ? "PASS" : "FAIL");
        if (!ok) {
            for (int e = 0; e < E; e++)
                std::printf("    expert %d: gpu=%d cpu=%d\n", e, gpu_start[e], cpu_start[e]);
            return false;
        }
    }

    // --- test 3: scatter kernel ---
    {
        int32_t* d_cursor;
        CHECK_CUDA(cudaMalloc(&d_cursor, E * sizeof(int32_t)));
        CHECK_CUDA(cudaMemset(d_cursor, 0, E * sizeof(int32_t)));
        CHECK_CUDA(cudaMemset(d_packed, 0, TK * d * sizeof(uint16_t)));

        int block = 32, grid = (TK + block - 1) / block;
        scatter_kernel<<<grid, block>>>(d_emb, d_asgn, d_expert_start, d_cursor,
                                        d_packed, d_perm, T, K, d);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<uint16_t> gpu_packed(TK * d);
        std::vector<int32_t>  gpu_perm(TK), gpu_start(E), gpu_count(E);
        CHECK_CUDA(cudaMemcpy(gpu_packed.data(), d_packed, TK * d * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_perm.data(),   d_perm,   TK * sizeof(int32_t),     cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_start.data(),  d_expert_start, E * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_count.data(),  d_expert_count, E * sizeof(int32_t), cudaMemcpyDeviceToHost));

        bool ok = verify_gpu_vs_cpu(
            h_emb.data(), h_asgn.data(),
            gpu_packed.data(), gpu_start.data(), gpu_count.data(), gpu_perm.data(),
            cpu_packed.data(), cpu_start.data(), cpu_count.data(), cpu_perm.data(),
            T, K, E, d);
        std::printf("  scatter_kernel:   %s\n", ok ? "PASS" : "FAIL");

        CHECK_CUDA(cudaFree(d_cursor));
        if (!ok) return false;
    }

    // --- test 4: full gpu_scatter pipeline ---
    {
        // re-upload fresh (scatter wrote to d_packed already, reset it)
        CHECK_CUDA(cudaMemset(d_packed, 0, TK * d * sizeof(uint16_t)));
        CHECK_CUDA(cudaMemset(d_expert_count, 0, E * sizeof(int32_t)));

        auto res = gpu_scatter(d_emb, d_asgn, T, K, E, d,
                               d_packed, d_expert_start, d_expert_count, d_perm);

        std::vector<uint16_t> gpu_packed(TK * d);
        std::vector<int32_t>  gpu_perm(TK), gpu_start(E), gpu_count(E);
        CHECK_CUDA(cudaMemcpy(gpu_packed.data(), d_packed, TK * d * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_perm.data(),   d_perm,   TK * sizeof(int32_t),     cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_start.data(),  d_expert_start, E * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_count.data(),  d_expert_count, E * sizeof(int32_t), cudaMemcpyDeviceToHost));

        bool ok = verify_gpu_vs_cpu(
            h_emb.data(), h_asgn.data(),
            gpu_packed.data(), gpu_start.data(), gpu_count.data(), gpu_perm.data(),
            cpu_packed.data(), cpu_start.data(), cpu_count.data(), cpu_perm.data(),
            T, K, E, d);
        std::printf("  full pipeline:    %s  (count=%.3fms prefix=%.3fms scatter=%.3fms)\n",
                    ok ? "PASS" : "FAIL", res.count_ms, res.prefix_ms, res.scatter_ms);
        if (!ok) return false;
    }

    // cleanup
    CHECK_CUDA(cudaFree(d_emb));  CHECK_CUDA(cudaFree(d_asgn));
    CHECK_CUDA(cudaFree(d_packed)); CHECK_CUDA(cudaFree(d_expert_start));
    CHECK_CUDA(cudaFree(d_expert_count)); CHECK_CUDA(cudaFree(d_perm));

    std::printf("\n  All mini-tests passed.\n");
    return true;
}

// ===========================================================================
// Full run: load .npy, run GPU scatter, verify against CPU, report timing
// ===========================================================================

static void run_full(const std::string& prefix, int E_cli, const std::string& data_dir) {
    std::printf("\n=== Full run: %s (E=%d) ===\n\n", prefix.c_str(), E_cli);

    NpyArray emb_npy  = load_npy(data_dir + "/" + prefix + "_embeddings.npy");
    NpyArray asgn_npy = load_npy(data_dir + "/" + prefix + "_assignments.npy");

    int T = (int)emb_npy.rows, d = (int)emb_npy.cols, K = (int)asgn_npy.cols;
    const uint16_t* h_emb  = (const uint16_t*)emb_npy.data.data();
    const int32_t*  h_asgn = (const int32_t*)asgn_npy.data.data();

    // figure out E
    int32_t max_e = -1;
    for (int i = 0; i < T * K; i++) if (h_asgn[i] > max_e) max_e = h_asgn[i];
    int E = std::max(E_cli, (int)(max_e + 1));

    int TK = T * K;
    std::printf("T=%d  K=%d  E=%d  d=%d  packed_rows=%d\n\n", T, K, E, d, TK);

    // CPU reference
    std::vector<uint16_t> cpu_packed(TK * d);
    std::vector<int32_t>  cpu_start(E), cpu_count(E), cpu_perm(TK);
    auto tc0 = std::chrono::steady_clock::now();
    cpu_scatter(h_emb, h_asgn, T, K, E, d,
                cpu_packed.data(), cpu_start.data(), cpu_count.data(), cpu_perm.data());
    auto tc1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();

    // GPU
    uint16_t *d_emb, *d_packed;
    int32_t  *d_asgn, *d_expert_start, *d_expert_count, *d_perm;
    CHECK_CUDA(cudaMalloc(&d_emb,          (size_t)T * d * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_asgn,         (size_t)TK * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_packed,       (size_t)TK * d * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_expert_start, E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_expert_count, E * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_perm,         (size_t)TK * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_emb,  h_emb,  (size_t)T * d * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_asgn, h_asgn, (size_t)TK * sizeof(int32_t),    cudaMemcpyHostToDevice));

    // warmup
    for (int i = 0; i < 3; i++) {
        gpu_scatter(d_emb, d_asgn, T, K, E, d,
                    d_packed, d_expert_start, d_expert_count, d_perm);
    }

    // timed run (average over 10 iterations)
    const int NRUNS = 10;
    GpuScatterResult total = {0, 0, 0};
    for (int i = 0; i < NRUNS; i++) {
        auto r = gpu_scatter(d_emb, d_asgn, T, K, E, d,
                             d_packed, d_expert_start, d_expert_count, d_perm);
        total.count_ms   += r.count_ms;
        total.prefix_ms  += r.prefix_ms;
        total.scatter_ms += r.scatter_ms;
    }
    float gpu_total = (total.count_ms + total.prefix_ms + total.scatter_ms) / NRUNS;

    // copy back for verification (last run's output)
    std::vector<uint16_t> gpu_packed(TK * d);
    std::vector<int32_t>  gpu_perm(TK), gpu_start(E), gpu_count(E);
    CHECK_CUDA(cudaMemcpy(gpu_packed.data(), d_packed, (size_t)TK * d * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gpu_perm.data(),   d_perm,   TK * sizeof(int32_t),              cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gpu_start.data(),  d_expert_start, E * sizeof(int32_t),          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gpu_count.data(),  d_expert_count, E * sizeof(int32_t),          cudaMemcpyDeviceToHost));

    bool ok = verify_gpu_vs_cpu(
        h_emb, h_asgn,
        gpu_packed.data(), gpu_start.data(), gpu_count.data(), gpu_perm.data(),
        cpu_packed.data(), cpu_start.data(), cpu_count.data(), cpu_perm.data(),
        T, K, E, d);

    size_t bytes_moved = (size_t)TK * d * sizeof(uint16_t);
    float avg_count   = total.count_ms / NRUNS;
    float avg_prefix  = total.prefix_ms / NRUNS;
    float avg_scatter = total.scatter_ms / NRUNS;

    std::printf("\nTiming (avg over %d runs):\n", NRUNS);
    std::printf("  count:    %8.3f ms\n", avg_count);
    std::printf("  prefix:   %8.3f ms\n", avg_prefix);
    std::printf("  scatter:  %8.3f ms\n", avg_scatter);
    std::printf("  total:    %8.3f ms  (%.2f GB/s effective)\n",
                gpu_total, (bytes_moved / 1e9) / (gpu_total / 1e3));
    std::printf("  cpu ref:  %8.3f ms\n", cpu_ms);
    std::printf("  speedup:  %.2fx\n", cpu_ms / gpu_total);

    // expert load summary
    int min_c = INT_MAX, max_c = 0;
    for (int e = 0; e < E; e++) {
        min_c = std::min(min_c, gpu_count[e]);
        max_c = std::max(max_c, gpu_count[e]);
    }
    double mean_c = (double)TK / E;
    std::printf("\nExpert load: min=%d  max=%d  mean=%.1f  imbalance=%.2fx\n",
                min_c, max_c, mean_c, max_c / mean_c);

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_emb));   CHECK_CUDA(cudaFree(d_asgn));
    CHECK_CUDA(cudaFree(d_packed)); CHECK_CUDA(cudaFree(d_expert_start));
    CHECK_CUDA(cudaFree(d_expert_count)); CHECK_CUDA(cudaFree(d_perm));
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
    // always run mini-tests first
    if (!run_mini_tests()) return 1;

    // if a prefix is given, also do a full run on .npy data
    if (argc > 1) {
        std::string prefix   = argv[1];
        int         E_cli    = (argc > 2) ? std::atoi(argv[2]) : 64;
        std::string data_dir = (argc > 3) ? argv[3] : "results";
        run_full(prefix, E_cli, data_dir);
    }

    return 0;
}
