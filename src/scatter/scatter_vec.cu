// src/scatter/scatter_vec.cu
//
// Strategy C: sort + warp-per-row gather with vectorized 16-byte loads.
//
// Identical to Strategy B (warp) except the gather kernel reads/writes the
// row in `uint4` (16 B = 8 fp16) chunks. d_model must be a multiple of 8.
//
// Why this helps on A100:
//   - Each warp instruction now moves 32 * 16 = 512 B (8 sectors of L2)
//     instead of 32 * 2 = 64 B (1 sector). 8x fewer LDG/STG instructions
//     per row, with the same total bytes — frees up issue slots and lets
//     us approach HBM bandwidth instead of being instruction-throughput
//     bound.
//   - Memory transactions per row drop from d/32 = 64 to d/(8*32) = 8.
//
// The sort + bounds + count kernels are unchanged from scatter_warp.cu.

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

// Single-block fused kernel: each thread does a binary search to find
// expert_start[e], then __syncthreads, then thread e (e<E) writes
// expert_count[e] = expert_start[e+1] - expert_start[e].
//
// Replaces the two-launch sequence (expert_bounds_kernel + expert_count_kernel)
// in scatter_warp.cu, saving one launch + one inter-kernel host gap.
//
// Constraint: must launch with grid=1, blockDim.x >= E+1. For typical MoE
// models E is 8/64/128/256, well under the 1024-thread block cap.
__global__ void expert_bounds_count_fused_kernel(
    const int32_t* __restrict__ sorted_keys,
    int32_t*       __restrict__ expert_start,
    int32_t*       __restrict__ expert_count,
    int TK, int E)
{
    int e = threadIdx.x;
    if (e > E) return;

    int my_start;
    if (e == E) {
        my_start = TK;
    } else {
        int lo = 0, hi = TK;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (sorted_keys[mid] < e) lo = mid + 1;
            else                      hi = mid;
        }
        my_start = lo;
    }
    expert_start[e] = my_start;

    __syncthreads();

    if (e < E) {
        expert_count[e] = expert_start[e + 1] - expert_start[e];
    }
}

constexpr int kWarpSize       = 32;
constexpr int kWarpsPerBlock  = 4;
constexpr int kVecPerLoad     = 8;       // 8 fp16 per uint4 = 16 bytes

// Warp-per-row gather, vectorized uint4 (16 B) loads/stores.
// d must be a multiple of kVecPerLoad (=8).
__global__ void gather_rows_vec_kernel(const __half*  __restrict__ emb,
                                       const int32_t* __restrict__ sorted_values,
                                       const float*   __restrict__ routing_wts,
                                       __half*        __restrict__ packed,
                                       float*         __restrict__ packed_wts,
                                       int32_t*       __restrict__ perm,
                                       int K, int d, int TK) {
    int warp_in_block = threadIdx.x / kWarpSize;
    int lane          = threadIdx.x & (kWarpSize - 1);
    int p             = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (p >= TK) return;

    int slot = sorted_values[p];
    int t    = slot / K;

    const uint4* src_v = reinterpret_cast<const uint4*>(emb    + (size_t)t * d);
    uint4*       dst_v = reinterpret_cast<uint4*>      (packed + (size_t)p * d);
    const int d_v = d / kVecPerLoad;
    for (int i = lane; i < d_v; i += kWarpSize) dst_v[i] = src_v[i];

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

class VecScatter final : public IScatter {
public:
    const char* name() const override { return "vec"; }

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

        if (p.d_model % kVecPerLoad != 0) {
            std::fprintf(stderr,
                "vec scatter requires d_model %% %d == 0 (got d=%d)\n",
                kVecPerLoad, p.d_model);
            std::exit(1);
        }

        SortWorkspace ws = plan_workspace(p);
        char* base = static_cast<char*>(bufs.strategy_ws);
        void*    d_cub        = base;
        int32_t* d_keys_out   = reinterpret_cast<int32_t*>(base + ws.keys_out_off);
        int32_t* d_values_in  = reinterpret_cast<int32_t*>(base + ws.values_in_off);
        int32_t* d_values_out = reinterpret_cast<int32_t*>(base + ws.values_out_off);

        if (p.E + 1 > 1024) {
            std::fprintf(stderr, "vec scatter: E+1=%d exceeds single-block "
                "fused-bounds limit (1024)\n", p.E + 1);
            std::exit(1);
        }

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

        // Fused single-launch bounds+count (untimed; stays in the gap between
        // sort_ms and gather_ms — same harness shape as scatter_warp.cu so
        // wall-clock comparison is apples-to-apples).
        {
            int bsz = 1;
            while (bsz < p.E + 1) bsz <<= 1;
            if (bsz < 64) bsz = 64;
            expert_bounds_count_fused_kernel<<<1, bsz, 0, stream>>>(
                d_keys_out, bufs.expert_start, bufs.expert_count, TK, p.E);
        }

        tg.start(stream);
        {
            const int threads = kWarpSize * kWarpsPerBlock;
            const int grd     = (TK + kWarpsPerBlock - 1) / kWarpsPerBlock;
            gather_rows_vec_kernel<<<grd, threads, 0, stream>>>(
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

std::unique_ptr<IScatter> make_vec_scatter() {
    return std::make_unique<VecScatter>();
}

} // namespace moe
