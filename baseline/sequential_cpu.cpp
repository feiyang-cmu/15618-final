// baseline/sequential_cpu.cpp
// ----------------------------------------------------------------------------
// Sequential CPU reference implementation of the MoE token packing (scatter)
// step. This is the ground-truth that every GPU strategy in strategies/ and
// fused/ must match bit-for-bit under verify/.
//
// Problem recap
// -------------
// Given:
//   embeddings   [T, d]    input hidden states (fp16/bf16 — opaque to us)
//   assignments  [T, K]    top-K expert IDs per token (int32)
//   E                      total number of experts
//
// Produce:
//   packed        [T*K, d] send buffer, rows grouped by expert ID
//   expert_start  [E]      exclusive prefix sum: row index where each
//                          expert's region starts inside `packed`
//   expert_count  [E]      number of rows assigned to each expert
//   permutation   [T*K]    permutation[pos] = the original token index `t`
//                          whose embedding was written to row `pos` of
//                          `packed`. The combine step walks this to return
//                          expert-FFN outputs back to original token order.
//
// The algorithm has four conceptual steps — counting, prefix sum, scatter,
// and verification. All are O(T*K + T*K*d) and fit in one sequential pass.
//
// Dtype note: the scatter step never touches the numerical values; it only
// copies d-wide rows around. We therefore treat the embedding payload as
// opaque uint16_t, which is correct for both fp16 and bf16.
//
// Build
// -----
//   g++ -O2 -std=c++17 baseline/sequential_cpu.cpp -o baseline/sequential_cpu
//
// Run
// ---
//   ./baseline/sequential_cpu                 # defaults to syn_uniform, E=64
//   ./baseline/sequential_cpu syn_zipf        # other synthetic distribution
//   ./baseline/sequential_cpu syn_uniform 64 results
// ----------------------------------------------------------------------------

#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Minimal .npy loader
// ----------------------------------------------------------------------------
// Only the subset used by this project: 2-D arrays, dtype '<f2' (little-endian
// float16) or '<i4' (little-endian int32). The host is assumed little-endian
// (x86-64), so the raw bytes can be used directly without byte swapping.

struct NpyArray {
    std::vector<uint8_t> data;
    size_t rows = 0;
    size_t cols = 0;
    std::string dtype;   // "<f2" or "<i4"
};

static NpyArray load_npy(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[npy] cannot open %s\n", path.c_str());
        std::exit(1);
    }

    // Magic number: 6 bytes "\x93NUMPY"
    char magic[6];
    if (std::fread(magic, 1, 6, fp) != 6 ||
        std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "[npy] bad magic in %s\n", path.c_str());
        std::exit(1);
    }

    auto must_read = [&](void* dst, size_t n) {
        if (std::fread(dst, 1, n, fp) != n) {
            std::fprintf(stderr, "[npy] short read in %s\n", path.c_str());
            std::exit(1);
        }
    };

    uint8_t major = 0, minor = 0;
    must_read(&major, 1);
    must_read(&minor, 1);

    // Header length: uint16 for v1, uint32 for v2+. All our files are v1.
    size_t header_len = 0;
    if (major == 1) {
        uint16_t h = 0;
        must_read(&h, 2);
        header_len = h;
    } else {
        uint32_t h = 0;
        must_read(&h, 4);
        header_len = h;
    }

    // Header is a Python-dict literal string. We do naive substring parsing
    // — good enough for the well-formed output of numpy.save().
    std::string header(header_len, ' ');
    must_read(&header[0], header_len);

    NpyArray arr;

    // dtype: look for 'descr': '<..'
    auto d_pos = header.find("'descr':");
    auto q1    = header.find('\'', d_pos + 8);
    auto q2    = header.find('\'', q1 + 1);
    arr.dtype  = header.substr(q1 + 1, q2 - q1 - 1);

    // shape: look for 'shape': (R, C)  — 2-D only
    auto s_pos = header.find("'shape':");
    auto lp    = header.find('(', s_pos);
    auto rp    = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    size_t comma = shape_str.find(',');
    arr.rows = std::strtoul(shape_str.substr(0, comma).c_str(), nullptr, 10);
    arr.cols = std::strtoul(shape_str.substr(comma + 1).c_str(), nullptr, 10);

    // Element size
    size_t elt = 0;
    if      (arr.dtype == "<f2") elt = 2;
    else if (arr.dtype == "<i4") elt = 4;
    else {
        std::fprintf(stderr, "[npy] unsupported dtype '%s' in %s\n",
                     arr.dtype.c_str(), path.c_str());
        std::exit(1);
    }

    size_t nbytes = arr.rows * arr.cols * elt;
    arr.data.resize(nbytes);
    if (std::fread(arr.data.data(), 1, nbytes, fp) != nbytes) {
        std::fprintf(stderr, "[npy] short read in %s\n", path.c_str());
        std::exit(1);
    }
    std::fclose(fp);

    std::printf("[npy] %-50s  shape=(%zu, %zu)  dtype=%s\n",
                path.c_str(), arr.rows, arr.cols, arr.dtype.c_str());
    return arr;
}

// ----------------------------------------------------------------------------
// cpu_scatter — sequential reference implementation
// ----------------------------------------------------------------------------
//
// Produces all four outputs in a single pass:
//   packed         [T*K, d]   expert-grouped send buffer
//   expert_start   [E]        per-expert write offset
//   expert_count   [E]        per-expert row count
//   permutation    [T*K]      packed-row -> original-token index
//
// The inner "copy d half-words" step dominates cost (T*K*d * 2 bytes moved),
// so the scatter is memory-bound even on CPU.
static void cpu_scatter(
    const uint16_t* embeddings,    // [T * d] row-major
    const int32_t*  assignments,   // [T * K] row-major
    int             T,
    int             K,
    int             E,
    int             d,
    uint16_t*       packed,        // [T*K * d] row-major, output
    int32_t*        expert_start,  // [E] output
    int32_t*        expert_count,  // [E] output
    int32_t*        permutation)   // [T*K] output
{
    // ------------------------------------------------------------------
    // Step 1 — Count how many rows each expert receives.
    //
    // Walk every (t, k) pair exactly once and increment the counter for
    // the chosen expert. Total increments across all experts = T * K,
    // because each token dispatches to exactly K experts.
    // ------------------------------------------------------------------
    std::memset(expert_count, 0, static_cast<size_t>(E) * sizeof(int32_t));
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            int e = assignments[t * K + k];
            expert_count[e] += 1;
        }
    }

    // ------------------------------------------------------------------
    // Step 2 — Exclusive prefix sum: expert_count -> expert_start.
    //
    // expert_start[e] is the row index inside `packed` where expert e's
    // region begins. expert_start[0] = 0, and each subsequent entry adds
    // the previous expert's count. After this loop, `running` equals
    // T * K (every (t, k) pair has a destination slot).
    // ------------------------------------------------------------------
    int32_t running = 0;
    for (int e = 0; e < E; ++e) {
        expert_start[e] = running;
        running += expert_count[e];
    }
    assert(running == T * K);

    // ------------------------------------------------------------------
    // Step 3 — Scatter: copy each token's d-wide row into its K slots.
    //
    // We keep a per-expert write cursor that starts at 0 and advances by
    // one every time a row is placed. The absolute destination row is
    //     pos = expert_start[e] + cursor[e]
    // Within an expert's region the row ordering is "first-come order from
    // the outer (t, k) loop" — any order is fine as long as we record it
    // in `permutation` so the combine step can invert it.
    //
    // Note: tokens assigned to the same expert multiple times would land
    // in adjacent slots here; the synthetic generators guarantee K unique
    // experts per token, so that case never occurs. The code still handles
    // it correctly regardless.
    // ------------------------------------------------------------------
    std::vector<int32_t> cursor(static_cast<size_t>(E), 0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            int e   = assignments[t * K + k];
            int pos = expert_start[e] + cursor[e];
            cursor[e] += 1;

            // Byte-copy one row of d half-words. memcpy is correct for
            // both fp16 and bf16 because no arithmetic is performed.
            std::memcpy(packed + static_cast<size_t>(pos) * d,
                        embeddings + static_cast<size_t>(t) * d,
                        static_cast<size_t>(d) * sizeof(uint16_t));

            // Record the inverse mapping for the combine step.
            permutation[pos] = t;
        }
    }

    // Sanity: every cursor must have advanced exactly expert_count[e] steps.
    for (int e = 0; e < E; ++e) {
        assert(cursor[e] == expert_count[e]);
    }
}

// ----------------------------------------------------------------------------
// verify_scatter — ground-truth self-check for cpu_scatter's output
// ----------------------------------------------------------------------------
//
// Catches the bug classes that matter for MoE packing:
//   (a) expert regions overlap or leave gaps
//   (b) a packed row's bytes don't match the embedding of its source token
//   (c) expert_count disagrees with a fresh recount from `assignments`
//   (d) a token appears a wrong number of times in `permutation`
//   (e) a packed row lives in the wrong expert's region
static bool verify_scatter(
    const uint16_t* embeddings,
    const int32_t*  assignments,
    const uint16_t* packed,
    const int32_t*  expert_start,
    const int32_t*  expert_count,
    const int32_t*  permutation,
    int             T,
    int             K,
    int             E,
    int             d)
{
    int errors = 0;
    const int MAX_PRINT = 20;

    // (a) Expert regions are contiguous and total to T*K.
    for (int e = 0; e + 1 < E; ++e) {
        if (expert_start[e + 1] != expert_start[e] + expert_count[e]) {
            if (errors < MAX_PRINT) {
                std::printf("  [err] gap at expert %d: start=%d count=%d next_start=%d\n",
                            e, expert_start[e], expert_count[e], expert_start[e + 1]);
            }
            ++errors;
        }
    }
    int32_t total = expert_start[E - 1] + expert_count[E - 1];
    if (total != T * K) {
        std::printf("  [err] total packed rows %d != T*K %d\n", total, T * K);
        ++errors;
    }

    // (b) Every packed row must equal the embedding of permutation[pos].
    for (int pos = 0; pos < T * K; ++pos) {
        int t = permutation[pos];
        if (t < 0 || t >= T) {
            if (errors < MAX_PRINT) {
                std::printf("  [err] permutation[%d] = %d out of [0, %d)\n", pos, t, T);
            }
            ++errors;
            continue;
        }
        if (std::memcmp(packed + static_cast<size_t>(pos) * d,
                        embeddings + static_cast<size_t>(t) * d,
                        static_cast<size_t>(d) * sizeof(uint16_t)) != 0) {
            if (errors < MAX_PRINT) {
                std::printf("  [err] packed row %d != embedding row %d\n", pos, t);
            }
            ++errors;
        }
    }

    // (c) Recount per-expert from `assignments` and compare.
    std::vector<int32_t> recount(static_cast<size_t>(E), 0);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
            recount[assignments[t * K + k]] += 1;
    for (int e = 0; e < E; ++e) {
        if (recount[e] != expert_count[e]) {
            if (errors < MAX_PRINT) {
                std::printf("  [err] expert %d count=%d recount=%d\n",
                            e, expert_count[e], recount[e]);
            }
            ++errors;
        }
    }

    // (d) Each token must appear exactly K times across `permutation`.
    std::vector<int32_t> times_seen(static_cast<size_t>(T), 0);
    for (int pos = 0; pos < T * K; ++pos) {
        int t = permutation[pos];
        if (t >= 0 && t < T) times_seen[t] += 1;
    }
    for (int t = 0; t < T; ++t) {
        if (times_seen[t] != K) {
            if (errors < MAX_PRINT) {
                std::printf("  [err] token %d appears %d times (expected %d)\n",
                            t, times_seen[t], K);
            }
            ++errors;
        }
    }

    // (e) Every row inside expert e's region must belong to a token that
    //     was actually assigned to expert e. (a)-(d) alone cannot catch a
    //     cross-expert swap that preserves per-expert counts.
    for (int e = 0; e < E && errors < MAX_PRINT + 100; ++e) {
        int lo = expert_start[e];
        int hi = lo + expert_count[e];
        for (int pos = lo; pos < hi; ++pos) {
            int t = permutation[pos];
            bool found = false;
            for (int k = 0; k < K; ++k) {
                if (assignments[t * K + k] == e) { found = true; break; }
            }
            if (!found) {
                if (errors < MAX_PRINT) {
                    std::printf("  [err] row %d of expert %d belongs to token %d "
                                "which was not assigned to %d\n", pos, e, t, e);
                }
                ++errors;
            }
        }
    }

    if (errors > MAX_PRINT) {
        std::printf("  ... %d additional errors suppressed\n", errors - MAX_PRINT);
    }
    return errors == 0;
}

// ----------------------------------------------------------------------------
// main: load .npy inputs, run cpu_scatter, verify, print a small report
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string prefix   = (argc > 1) ? argv[1] : "syn_uniform";
    int         E_cli    = (argc > 2) ? std::atoi(argv[2]) : 64;
    std::string data_dir = (argc > 3) ? argv[3] : "results";

    std::printf("== Sequential CPU scatter (ground-truth) ==\n");
    std::printf("   prefix='%s'  data_dir='%s'  E(cli)=%d\n\n",
                prefix.c_str(), data_dir.c_str(), E_cli);

    // Load inputs from the synthetic routing .npy files generated by
    // data/gen_synthetic_routing.py. `router_weights` is not required by
    // the scatter step itself and is left for the combine step.
    NpyArray emb  = load_npy(data_dir + "/" + prefix + "_embeddings.npy");
    NpyArray asgn = load_npy(data_dir + "/" + prefix + "_assignments.npy");

    assert(emb.dtype  == "<f2");
    assert(asgn.dtype == "<i4");
    assert(emb.rows   == asgn.rows);

    int T = static_cast<int>(emb.rows);
    int d = static_cast<int>(emb.cols);
    int K = static_cast<int>(asgn.cols);

    // The .npy files do not carry E explicitly. Use the CLI value, but
    // widen it if we observe a larger expert ID in the data.
    const int32_t* asgn_ptr = reinterpret_cast<const int32_t*>(asgn.data.data());
    int32_t max_e = -1;
    for (size_t i = 0; i < asgn.rows * asgn.cols; ++i)
        if (asgn_ptr[i] > max_e) max_e = asgn_ptr[i];
    int E = E_cli;
    if (max_e + 1 > E) E = max_e + 1;

    std::printf("\nDims: T=%d  K=%d  E=%d  d=%d  (packed rows = %d)\n\n",
                T, K, E, d, T * K);

    // Allocate outputs.
    std::vector<uint16_t> packed(static_cast<size_t>(T) * K * d);
    std::vector<int32_t>  expert_start(static_cast<size_t>(E));
    std::vector<int32_t>  expert_count(static_cast<size_t>(E));
    std::vector<int32_t>  permutation(static_cast<size_t>(T) * K);

    // Run and time the scatter.
    const uint16_t* emb_ptr = reinterpret_cast<const uint16_t*>(emb.data.data());
    auto t0 = std::chrono::steady_clock::now();
    cpu_scatter(emb_ptr, asgn_ptr, T, K, E, d,
                packed.data(), expert_start.data(),
                expert_count.data(), permutation.data());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t bytes_moved = static_cast<size_t>(T) * K * d * sizeof(uint16_t);
    double gbps = (bytes_moved / 1e9) / (ms / 1e3);
    std::printf("cpu_scatter: %.3f ms   (%.2f GB/s effective, %.2f MB moved)\n",
                ms, gbps, bytes_moved / (1024.0 * 1024.0));

    // Verify.
    bool ok = verify_scatter(emb_ptr, asgn_ptr,
                             packed.data(), expert_start.data(),
                             expert_count.data(), permutation.data(),
                             T, K, E, d);

    // Expert-load summary (matches the stats the Python generator prints).
    int min_c = INT_MAX, max_c = 0;
    long long sum_c = 0;
    for (int e = 0; e < E; ++e) {
        if (expert_count[e] < min_c) min_c = expert_count[e];
        if (expert_count[e] > max_c) max_c = expert_count[e];
        sum_c += expert_count[e];
    }
    double mean_c = static_cast<double>(sum_c) / E;
    std::printf("Expert load: min=%d  max=%d  mean=%.1f  total=%lld  imbalance=%.2fx\n",
                min_c, max_c, mean_c, sum_c,
                mean_c > 0 ? max_c / mean_c : 0.0);

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
