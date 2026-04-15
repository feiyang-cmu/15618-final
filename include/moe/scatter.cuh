// include/moe/scatter.cuh
//
// IScatter: strategy-polymorphic interface for the MoE token-packing step.
// A scatter reads (embeddings, assignments, routing_weights) and fills the
// scatter outputs inside MoEBuffers. Strategies (atomic / sort / warp) differ
// only in how they achieve that mapping.

#pragma once

#include "types.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <string>

namespace moe {

/// Break-down of per-stage timings returned by IScatter::run.
/// Fields are strategy-dependent; zero means "not applicable".
struct ScatterTimings {
    float count_ms  = 0.f;
    float prefix_ms = 0.f;
    float sort_ms   = 0.f;
    float gather_ms = 0.f;
    float total_ms  = 0.f;
};

class IScatter {
public:
    virtual ~IScatter() = default;

    /// Human-readable strategy name, e.g. "atomic", "sort", "warp".
    virtual const char* name() const = 0;

    /// Bytes of scratch the strategy needs. Caller allocates and sets
    /// MoEBuffers::strategy_ws / strategy_ws_bytes before calling run().
    virtual std::size_t workspace_bytes(const MoEParams& p) const = 0;

    /// Execute the scatter.
    ///
    /// Inputs (all device pointers):
    ///   d_emb     [T, d_model]   fp16 token embeddings
    ///   d_asgn    [T, K]         int32  expert index per (token, k)
    ///   d_wts     [T, K]         fp32   routing weight per (token, k)
    ///
    /// Outputs written into bufs:
    ///   packed, packed_weights, perm, expert_count, expert_start
    virtual ScatterTimings run(const __half*  d_emb,
                               const int32_t* d_asgn,
                               const float*   d_wts,
                               MoEBuffers&    bufs,
                               const MoEParams& p,
                               cudaStream_t   stream) = 0;
};

// Per-strategy constructors. Each strategy defines exactly one of these.
std::unique_ptr<IScatter> make_atomic_scatter();
std::unique_ptr<IScatter> make_sort_scatter();

/// Factory. Accepted names: "atomic", "sort" (Phase 2), "warp" (Phase 3).
/// Returns nullptr for unknown names.
std::unique_ptr<IScatter> make_scatter(const std::string& name);

} // namespace moe
