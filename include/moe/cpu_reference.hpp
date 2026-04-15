// include/moe/cpu_reference.hpp
//
// Ground-truth CPU implementation of the scatter step, plus an
// order-independent invariant checker. Any IScatter implementation whose
// output satisfies verify_scatter_outputs() against the same inputs is
// functionally equivalent to the reference — this sidesteps the fact that
// atomic/warp strategies produce non-deterministic row orderings within each
// expert region.

#pragma once

#include <cstdint>

namespace moe {

/// Embeddings are treated as opaque 16-bit words; the scatter never performs
/// arithmetic on them. uint16_t is correct for both fp16 and bf16 payloads.
void cpu_scatter(const std::uint16_t* embeddings,      // [T, d]
                 const std::int32_t*  assignments,     // [T, K]
                 const float*         routing_weights, // [T, K], may be nullptr
                 int T, int K, int E, int d,
                 std::uint16_t* packed,         // [T*K, d]
                 float*         packed_weights, // [T*K],    may be nullptr iff routing_weights is
                 std::int32_t*  expert_start,   // [E]
                 std::int32_t*  expert_count,   // [E]
                 std::int32_t*  perm);          // [T*K]

/// Checks invariants that any correct scatter output must satisfy:
///   (a) expert regions are contiguous and total T*K
///   (b) every packed row equals embeddings[perm[pos]]
///   (c) expert_count matches a fresh recount of `assignments`
///   (d) every token appears exactly K times in `perm`
///   (e) every row in expert e's region is owned by a token assigned to e
///   (f) if routing_weights is provided, packed_weights[pos] matches the
///       routing weight for (perm[pos], k) where k is the slot that dispatched
///       to expert e (the expert of the region that contains pos)
/// Returns true on success; prints up to 20 failures on mismatch.
bool verify_scatter_outputs(const std::uint16_t* embeddings,
                            const std::int32_t*  assignments,
                            const float*         routing_weights,   // may be nullptr
                            const std::uint16_t* packed,
                            const float*         packed_weights,    // may be nullptr
                            const std::int32_t*  expert_start,
                            const std::int32_t*  expert_count,
                            const std::int32_t*  perm,
                            int T, int K, int E, int d);

} // namespace moe
