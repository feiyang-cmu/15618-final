// src/common/cpu_reference.cpp

#include "moe/cpu_reference.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

namespace moe {

void cpu_scatter(const std::uint16_t* embeddings,
                 const std::int32_t*  assignments,
                 const float*         routing_weights,
                 int T, int K, int E, int d,
                 std::uint16_t* packed,
                 float*         packed_weights,
                 std::int32_t*  expert_start,
                 std::int32_t*  expert_count,
                 std::int32_t*  perm) {
    // Step 1 — count
    std::memset(expert_count, 0, (std::size_t)E * sizeof(std::int32_t));
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
            expert_count[assignments[t * K + k]] += 1;

    // Step 2 — exclusive prefix sum
    std::int32_t running = 0;
    for (int e = 0; e < E; ++e) {
        expert_start[e] = running;
        running += expert_count[e];
    }
    assert(running == T * K);

    // Step 3 — scatter
    std::vector<std::int32_t> cursor((std::size_t)E, 0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            int slot = t * K + k;
            int e    = assignments[slot];
            int pos  = expert_start[e] + cursor[e]++;
            std::memcpy(packed + (std::size_t)pos * d,
                        embeddings + (std::size_t)t * d,
                        (std::size_t)d * sizeof(std::uint16_t));
            if (packed_weights && routing_weights) {
                packed_weights[pos] = routing_weights[slot];
            }
            perm[pos] = t;
        }
    }
}

// ----------------------------------------------------------------------------

namespace {

constexpr int MAX_PRINT = 20;

} // namespace

bool verify_scatter_outputs(const std::uint16_t* embeddings,
                            const std::int32_t*  assignments,
                            const float*         routing_weights,
                            const std::uint16_t* packed,
                            const float*         packed_weights,
                            const std::int32_t*  expert_start,
                            const std::int32_t*  expert_count,
                            const std::int32_t*  perm,
                            int T, int K, int E, int d) {
    int errors = 0;
    const int TK = T * K;

    // (a) contiguous regions summing to T*K
    for (int e = 0; e + 1 < E; ++e) {
        if (expert_start[e + 1] != expert_start[e] + expert_count[e]) {
            if (errors < MAX_PRINT)
                std::printf("  [err] gap at expert %d: start=%d count=%d next=%d\n",
                            e, expert_start[e], expert_count[e], expert_start[e + 1]);
            ++errors;
        }
    }
    std::int32_t total = expert_start[E - 1] + expert_count[E - 1];
    if (total != TK) {
        std::printf("  [err] total packed rows %d != T*K %d\n", total, TK);
        ++errors;
    }

    // (b) packed row matches embeddings[perm[pos]]
    for (int pos = 0; pos < TK; ++pos) {
        int t = perm[pos];
        if (t < 0 || t >= T) {
            if (errors < MAX_PRINT)
                std::printf("  [err] perm[%d]=%d out of [0,%d)\n", pos, t, T);
            ++errors; continue;
        }
        if (std::memcmp(packed + (std::size_t)pos * d,
                        embeddings + (std::size_t)t * d,
                        (std::size_t)d * sizeof(std::uint16_t)) != 0) {
            if (errors < MAX_PRINT)
                std::printf("  [err] packed row %d != embedding row %d\n", pos, t);
            ++errors;
        }
    }

    // (c) expert_count matches recount of assignments
    std::vector<std::int32_t> recount((std::size_t)E, 0);
    for (int i = 0; i < TK; ++i) recount[assignments[i]] += 1;
    for (int e = 0; e < E; ++e) {
        if (recount[e] != expert_count[e]) {
            if (errors < MAX_PRINT)
                std::printf("  [err] expert %d count=%d recount=%d\n",
                            e, expert_count[e], recount[e]);
            ++errors;
        }
    }

    // (d) every token appears exactly K times in perm
    std::vector<std::int32_t> seen((std::size_t)T, 0);
    for (int pos = 0; pos < TK; ++pos) {
        int t = perm[pos];
        if (t >= 0 && t < T) seen[t] += 1;
    }
    for (int t = 0; t < T; ++t) {
        if (seen[t] != K) {
            if (errors < MAX_PRINT)
                std::printf("  [err] token %d appears %d times (expected %d)\n",
                            t, seen[t], K);
            ++errors;
        }
    }

    // (e) each row in expert e's region is owned by a token assigned to e;
    // (f) if routing_weights given, packed_weights matches the routing weight
    //     for that (token, k) slot.
    //
    // We walk each (t,k) slot from the input and mark it "used" once we match
    // it to a packed row. That avoids double-counting when the same token
    // appears in expert e's region multiple times (possible in principle).
    std::vector<char> slot_used((std::size_t)TK, 0);
    for (int e = 0; e < E && errors < MAX_PRINT + 100; ++e) {
        int lo = expert_start[e];
        int hi = lo + expert_count[e];
        for (int pos = lo; pos < hi; ++pos) {
            int t = perm[pos];
            int matched_k = -1;
            for (int k = 0; k < K; ++k) {
                int slot = t * K + k;
                if (assignments[slot] == e && !slot_used[slot]) {
                    slot_used[slot] = 1;
                    matched_k = k;
                    break;
                }
            }
            if (matched_k < 0) {
                if (errors < MAX_PRINT)
                    std::printf("  [err] row %d of expert %d: no unused (t=%d, k=*) slot\n",
                                pos, e, t);
                ++errors;
                continue;
            }
            if (routing_weights && packed_weights) {
                float expect = routing_weights[t * K + matched_k];
                float got    = packed_weights[pos];
                if (expect != got) {
                    if (errors < MAX_PRINT)
                        std::printf("  [err] weight mismatch pos=%d: expect=%.6g got=%.6g\n",
                                    pos, expect, got);
                    ++errors;
                }
            }
        }
    }

    if (errors > MAX_PRINT) {
        std::printf("  ... %d additional errors suppressed\n", errors - MAX_PRINT);
    }
    return errors == 0;
}

} // namespace moe
