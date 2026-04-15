// src/scatter/scatter_sort.cu
//
// Strategy A: sort-based scatter.
//
// Pipeline (4 kernels + 1 CUB call):
//   1. init_values_kernel  : values_in[i] = i, for i in [0, TK)
//   2. cub::DeviceRadixSort::SortPairs
//        keys   = assignments   (expert ids, range [0, E))
//        values = 0..TK-1       (original slot indices)
//        Only log2(E) bits are needed — cap end_bit for speed.
//   3. expert_bounds_kernel : E threads, each binary-searches sorted_keys
//                             to find its lower_bound, filling expert_start.
//                             expert_count[e] = expert_start[e+1]-expert_start[e].
//   4. gather_rows_kernel  : grid = TK, one block per output slot p.
//                             Thread 0 reads sorted_values[p] = original slot,
//                             derives t = slot/K, then the whole block
//                             copies emb[t, :] -> packed[p, :] in a fully
//                             coalesced memcpy pattern (striding blockDim.x
//                             across d_model).
//
// Why this beats atomic on skewed distributions:
//   - No per-expert cursor atomic contention (sort decides position).
//   - Gather becomes fully coalesced (one block per output row).
//   - Tail latency flattens under Zipf because no expert is a hot counter.

#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe {
namespace {

__global__ void init_values_kernel(int32_t* values, int TK) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < TK) values[i] = i;
}

// Fill expert_start[0..E] and expert_count[0..E-1] from sorted expert ids.
// Each thread owns one expert e, binary-searches the lower_bound of e in
// sorted_keys to get expert_start[e]. Thread 0 also writes expert_start[E]=TK,
// after which a trivial diff gives expert_count[e] = start[e+1]-start[e].
__global__ void expert_bounds_kernel(const int32_t* __restrict__ sorted_keys,
                                     int32_t*       __restrict__ expert_start,
                                     int32_t*       __restrict__ expert_count,
                                     int TK, int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e > E) return;

    if (e == E) {
        expert_start[E] = TK;
        return;
    }

    int lo = 0, hi = TK;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (sorted_keys[mid] < e) lo = mid + 1;
        else                      hi = mid;
    }
    expert_start[e] = lo;
}

__global__ void expert_count_kernel(const int32_t* __restrict__ expert_start,
                                    int32_t*       __restrict__ expert_count,
                                    int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    expert_count[e] = expert_start[e + 1] - expert_start[e];
}

// One block per output slot. Block threads cooperatively copy d_model fp16
// elements from the source token row into the packed slot row. Block size is
// chosen by the launcher (256 works for all our configs; d_model % 256 == 0
// for Qwen-like shapes).
__global__ void gather_rows_kernel(const __half*  __restrict__ emb,          // [T, d]
                                   const int32_t* __restrict__ sorted_values,// [TK]
                                   const float*   __restrict__ routing_wts,  // [T*K]
                                   __half*        __restrict__ packed,       // [TK, d]
                                   float*         __restrict__ packed_wts,   // [TK]
                                   int32_t*       __restrict__ perm,         // [TK]
                                   int K, int d) {
    int p    = blockIdx.x;
    int slot = sorted_values[p];
    int t    = slot / K;

    const __half* src = emb    + (size_t)t * d;
    __half*       dst = packed + (size_t)p * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];

    if (threadIdx.x == 0) {
        packed_wts[p] = routing_wts[slot];
        perm[p]       = t;
    }
}

constexpr std::size_t kAlign = 16;
static inline std::size_t align_up(std::size_t x) {
    return (x + kAlign - 1) & ~(kAlign - 1);
}
static inline int ceil_log2(int x) {
    int b = 0;
    while ((1 << b) < x) ++b;
    return b;
}

// Workspace layout (in order):
//   [cub_bytes) CUB temp
//   [4*TK)      keys_out     (sorted expert ids)
//   [4*TK)      values_in    (0..TK-1 input)
//   [4*TK)      values_out   (sorted original slot indices)
struct SortWorkspace {
    std::size_t cub_bytes;
    std::size_t keys_out_off;
    std::size_t values_in_off;
    std::size_t values_out_off;
    std::size_t total_bytes;
};

SortWorkspace plan_workspace(const MoEParams& p) {
    SortWorkspace ws{};
    const int TK      = p.T * p.K;
    const int end_bit = ceil_log2(std::max(2, p.E));

    // CUB asks for temp storage given problem size & bit range.
    cub::DeviceRadixSort::SortPairs(
        nullptr, ws.cub_bytes,
        (const int32_t*)nullptr, (int32_t*)nullptr,
        (const int32_t*)nullptr, (int32_t*)nullptr,
        TK, 0, end_bit);

    ws.cub_bytes       = align_up(ws.cub_bytes);
    ws.keys_out_off    = ws.cub_bytes;
    ws.values_in_off   = ws.keys_out_off   + align_up((std::size_t)TK * sizeof(int32_t));
    ws.values_out_off  = ws.values_in_off  + align_up((std::size_t)TK * sizeof(int32_t));
    ws.total_bytes     = ws.values_out_off + align_up((std::size_t)TK * sizeof(int32_t));
    return ws;
}

class SortScatter final : public IScatter {
public:
    const char* name() const override { return "sort"; }

    std::size_t workspace_bytes(const MoEParams& p) const override {
        return plan_workspace(p).total_bytes;
    }

    ScatterTimings run(const __half*  d_emb,
                       const int32_t* d_asgn,
                       const float*   d_wts,
                       MoEBuffers&    bufs,
                       const MoEParams& p,
                       cudaStream_t   stream) override {
        const int TK      = p.T * p.K;
        const int end_bit = ceil_log2(std::max(2, p.E));

        SortWorkspace ws = plan_workspace(p);
        char* base = static_cast<char*>(bufs.strategy_ws);
        void*    d_cub        = base;
        int32_t* d_keys_out   = reinterpret_cast<int32_t*>(base + ws.keys_out_off);
        int32_t* d_values_in  = reinterpret_cast<int32_t*>(base + ws.values_in_off);
        int32_t* d_values_out = reinterpret_cast<int32_t*>(base + ws.values_out_off);

        ScatterTimings t;
        CudaTimer tc, ts, tg, tt;
        tt.start(stream);

        // Step 1: init values_in = 0..TK-1 (one tiny kernel). We also reuse
        // `tc` as "prep" time — CUB SortPairs dominates anyway.
        tc.start(stream);
        {
            const int blk = 256, grd = (TK + blk - 1) / blk;
            init_values_kernel<<<grd, blk, 0, stream>>>(d_values_in, TK);
        }
        t.count_ms = tc.stop_ms(stream);

        // Step 2: CUB radix sort. Keys are a const view of d_asgn.
        ts.start(stream);
        cub::DeviceRadixSort::SortPairs(
            d_cub, ws.cub_bytes,
            d_asgn, d_keys_out,
            d_values_in, d_values_out,
            TK, 0, end_bit, stream);
        t.sort_ms = ts.stop_ms(stream);

        // Step 3: compute expert_start[0..E] via E+1 parallel binary searches,
        // then expert_count[e] = start[e+1]-start[e].
        {
            const int blk = 128;
            const int grd = (p.E + 1 + blk - 1) / blk;
            expert_bounds_kernel<<<grd, blk, 0, stream>>>(
                d_keys_out, bufs.expert_start, bufs.expert_count, TK, p.E);
            const int grd2 = (p.E + blk - 1) / blk;
            expert_count_kernel<<<grd2, blk, 0, stream>>>(
                bufs.expert_start, bufs.expert_count, p.E);
        }

        // Step 4: one block per output slot; coalesced row copy.
        tg.start(stream);
        {
            const int blk = 256;
            gather_rows_kernel<<<TK, blk, 0, stream>>>(
                d_emb, d_values_out, d_wts,
                bufs.packed, bufs.packed_weights, bufs.perm,
                p.K, p.d_model);
        }
        t.gather_ms = tg.stop_ms(stream);

        t.total_ms = tt.stop_ms(stream);
        return t;
    }
};

} // namespace

std::unique_ptr<IScatter> make_sort_scatter() {
    return std::make_unique<SortScatter>();
}

} // namespace moe
