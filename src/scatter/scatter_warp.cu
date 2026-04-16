// src/scatter/scatter_warp.cu
//
// Strategy B: sort + warp-per-row gather.
//
// Identical to Strategy A (sort) except for the gather kernel geometry:
//   Sort uses one *block* per output row; threads stride d_model.
//   Warp  uses one *warp*  per output row; lanes stride d_model.
//
// Motivation: at small d_model, a 256-thread block striding d elements
// leaves (256 - d) threads idle on the row copy. A warp-per-row variant
// keeps every lane busy and packs more rows into the same grid, so each
// block launch amortizes over multiple copies. At large d_model (our
// main bench shape, d=2048) the block variant wins because it exposes
// more parallelism per row; the crossover is the interesting data point.
//
// The sort / bounds / count kernels are unchanged from scatter_sort.cu.
// A shared helper would let them collapse into one TU, but keeping
// strategies independent (matching the factory model) is worth the
// duplication for a 60-line prefix.

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

__global__ void expert_bounds_kernel(const int32_t* __restrict__ sorted_keys,
                                     int32_t*       __restrict__ expert_start,
                                     int32_t*       __restrict__ expert_count,
                                     int TK, int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e > E) return;
    if (e == E) { expert_start[E] = TK; return; }

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

// One WARP per output row. Each block holds WARPS_PER_BLOCK warps, so grid
// shrinks by that factor versus the sort strategy's grid-per-row launch.
// Lanes stride 32-wide across d_model. Lane 0 also writes the scalar
// packed_weights / perm slots.
constexpr int kWarpSize       = 32;
constexpr int kWarpsPerBlock  = 4;      // 128 threads/block

__global__ void gather_rows_warp_kernel(const __half*  __restrict__ emb,          // [T, d]
                                        const int32_t* __restrict__ sorted_values,// [TK]
                                        const float*   __restrict__ routing_wts,  // [T*K]
                                        __half*        __restrict__ packed,       // [TK, d]
                                        float*         __restrict__ packed_wts,   // [TK]
                                        int32_t*       __restrict__ perm,         // [TK]
                                        int K, int d, int TK) {
    int warp_in_block = threadIdx.x / kWarpSize;
    int lane          = threadIdx.x & (kWarpSize - 1);
    int p             = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (p >= TK) return;

    int slot = sorted_values[p];
    int t    = slot / K;

    const __half* src = emb    + (size_t)t * d;
    __half*       dst = packed + (size_t)p * d;
    for (int i = lane; i < d; i += kWarpSize) dst[i] = src[i];

    if (lane == 0) {
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

class WarpScatter final : public IScatter {
public:
    const char* name() const override { return "warp"; }

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

        tc.start(stream);
        {
            const int blk = 256, grd = (TK + blk - 1) / blk;
            init_values_kernel<<<grd, blk, 0, stream>>>(d_values_in, TK);
        }
        t.count_ms = tc.stop_ms(stream);

        ts.start(stream);
        cub::DeviceRadixSort::SortPairs(
            d_cub, ws.cub_bytes,
            d_asgn, d_keys_out,
            d_values_in, d_values_out,
            TK, 0, end_bit, stream);
        t.sort_ms = ts.stop_ms(stream);

        {
            const int blk = 128;
            const int grd = (p.E + 1 + blk - 1) / blk;
            expert_bounds_kernel<<<grd, blk, 0, stream>>>(
                d_keys_out, bufs.expert_start, bufs.expert_count, TK, p.E);
            const int grd2 = (p.E + blk - 1) / blk;
            expert_count_kernel<<<grd2, blk, 0, stream>>>(
                bufs.expert_start, bufs.expert_count, p.E);
        }

        tg.start(stream);
        {
            const int threads = kWarpSize * kWarpsPerBlock;     // 128
            const int grd     = (TK + kWarpsPerBlock - 1) / kWarpsPerBlock;
            gather_rows_warp_kernel<<<grd, threads, 0, stream>>>(
                d_emb, d_values_out, d_wts,
                bufs.packed, bufs.packed_weights, bufs.perm,
                p.K, p.d_model, TK);
        }
        t.gather_ms = tg.stop_ms(stream);

        t.total_ms = tt.stop_ms(stream);
        return t;
    }
};

} // namespace

std::unique_ptr<IScatter> make_warp_scatter() {
    return std::make_unique<WarpScatter>();
}

} // namespace moe
