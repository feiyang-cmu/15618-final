// src/scatter/scatter_atomic.cu
//
// Baseline "atomic" scatter strategy.
//
// Pipeline (3 kernels + 1 CUB call):
//   1. count_kernel     : atomicAdd into expert_count[E]
//   2. cub::ExclusiveSum: expert_count -> expert_start
//   3. scatter_kernel   : atomicAdd on per-expert cursor to claim a slot,
//                         then copy the token row and the routing weight
//
// Known non-optimality: step 1 and step 3 both serialize on hot experts
// under skewed (Zipf) distributions. This is the contention that Strategy A
// (sort-based) is designed to eliminate.

#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace moe {
namespace {

// Block-local histogram into shared memory, then one global atomicAdd per
// (block, expert). For small E this eliminates the global-atomic hotspot that
// plagues the naive per-thread atomicAdd implementation — the old e2e code
// side-stepped this by folding count into topk_kernel, which isn't possible
// once counting lives behind the IScatter boundary.
__global__ void count_kernel(const int32_t* __restrict__ assignments,
                             int32_t*       __restrict__ expert_count,
                             int TK, int E) {
    extern __shared__ int32_t s_hist[];  // [E]
    for (int i = threadIdx.x; i < E; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TK) atomicAdd(&s_hist[assignments[idx]], 1);
    __syncthreads();

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        int32_t v = s_hist[i];
        if (v) atomicAdd(&expert_count[i], v);
    }
}

__global__ void scatter_kernel(const __half*  __restrict__ emb,          // [T, d]
                               const int32_t* __restrict__ assignments,  // [T, K]
                               const float*   __restrict__ routing_wts,  // [T, K]
                               const int32_t* __restrict__ expert_start, // [E]
                               int32_t*       __restrict__ cursor,       // [E], zeroed
                               __half*        __restrict__ packed,       // [T*K, d]
                               float*         __restrict__ packed_wts,   // [T*K]
                               int32_t*       __restrict__ perm,         // [T*K]
                               int T, int K, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int TK  = T * K;
    if (idx >= TK) return;

    int t = idx / K;
    int e = assignments[idx];

    int slot = atomicAdd(&cursor[e], 1);
    int pos  = expert_start[e] + slot;

    const __half* src = emb    + (size_t)t   * d;
    __half*       dst = packed + (size_t)pos * d;
    for (int i = 0; i < d; ++i) dst[i] = src[i];

    packed_wts[pos] = routing_wts[idx];
    perm[pos]       = t;
}

/// Workspace layout:
///   [0 .. cub_tmp_bytes)                      : CUB temp storage
///   [cub_tmp_bytes .. +E*sizeof(int32_t))     : per-expert cursor
struct AtomicWorkspace {
    std::size_t cub_bytes;
    std::size_t total_bytes;
};

constexpr std::size_t kAlign = 16;

AtomicWorkspace plan_workspace(const MoEParams& p) {
    AtomicWorkspace ws{};
    cub::DeviceScan::ExclusiveSum<int32_t*, int32_t*>(
        nullptr, ws.cub_bytes,
        (int32_t*)nullptr, (int32_t*)nullptr, p.E);
    // CUB does not promise any particular size — round up so the cursor
    // allocation that immediately follows is naturally aligned.
    ws.cub_bytes  = (ws.cub_bytes + kAlign - 1) & ~(kAlign - 1);
    ws.total_bytes = ws.cub_bytes + (std::size_t)p.E * sizeof(int32_t);
    return ws;
}

class AtomicScatter final : public IScatter {
public:
    const char* name() const override { return "atomic"; }

    std::size_t workspace_bytes(const MoEParams& p) const override {
        return plan_workspace(p).total_bytes;
    }

    ScatterTimings run(const __half*  d_emb,
                       const int32_t* d_asgn,
                       const float*   d_wts,
                       MoEBuffers&    bufs,
                       const MoEParams& p,
                       cudaStream_t   stream) override {
        const int TK    = p.T * p.K;
        const int block = 256;
        const int grid  = (TK + block - 1) / block;

        AtomicWorkspace ws = plan_workspace(p);
        void*    d_cub_tmp = bufs.strategy_ws;
        int32_t* d_cursor  = reinterpret_cast<int32_t*>(
            static_cast<char*>(bufs.strategy_ws) + ws.cub_bytes);

        ScatterTimings t;
        CudaTimer tc, tp, ts, tt;

        tt.start(stream);

        // Step 1: count.
        device_zero_async(bufs.expert_count, (std::size_t)p.E, stream);
        tc.start(stream);
        count_kernel<<<grid, block, p.E * sizeof(int32_t), stream>>>(
            d_asgn, bufs.expert_count, TK, p.E);
        t.count_ms = tc.stop_ms(stream);

        // Step 2: exclusive prefix sum.
        tp.start(stream);
        cub::DeviceScan::ExclusiveSum(d_cub_tmp, ws.cub_bytes,
                                      bufs.expert_count,
                                      bufs.expert_start,
                                      p.E, stream);
        t.prefix_ms = tp.stop_ms(stream);

        // Step 3: scatter.
        device_zero_async(d_cursor, (std::size_t)p.E, stream);
        ts.start(stream);
        scatter_kernel<<<grid, block, 0, stream>>>(
            d_emb, d_asgn, d_wts,
            bufs.expert_start, d_cursor,
            bufs.packed, bufs.packed_weights, bufs.perm,
            p.T, p.K, p.d_model);
        t.gather_ms = ts.stop_ms(stream);

        t.total_ms = tt.stop_ms(stream);
        return t;
    }
};

} // namespace

std::unique_ptr<IScatter> make_scatter(const std::string& name) {
    if (name == "atomic") return std::make_unique<AtomicScatter>();
    // "sort" and "warp" are added on their respective branches.
    return nullptr;
}

} // namespace moe
