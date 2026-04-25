// src/scatter/scatter_csort.cu
//
// Strategy D: counting-sort + warp-per-row vectorized gather (single fused pass).
//
// Pipeline (3 ops; no CUB radix sort, no separate bounds kernels):
//   1. count_kernel        : shmem-histogram → expert_count    (same as atomic)
//   2. cub::ExclusiveSum   : expert_count    → expert_start    (~5 µs, ~64 ints)
//   3. place_gather_kernel : warp-per-row, lane 0 atomically reserves a slot
//                            within its expert, all lanes copy the row in
//                            uint4 (16 B) chunks
//
// Replaces in scatter_warp.cu: init_values_kernel, CUB DeviceRadixSort::SortPairs,
// expert_bounds_kernel, expert_count_kernel — i.e. ~50–80 µs of launch + sort
// overhead in the warp pipeline (~38% of warp-strategy total).
//
// Hot-expert atomic contention under Zipf: lane 0 of each warp does one
// atomicAdd(cursor[e], 1). For the hottest expert with ~1590 tokens that's
// ~1590 atomics on the same address — which on A100 serializes at the L2-atomic
// throughput of ~1 op/cycle, so ~1.6 µs total. Negligible vs the savings.

#include "moe/scatter.cuh"
#include "moe/types.h"
#include "moe/utils.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cstdint>

namespace moe {
namespace {

// Reused from scatter_atomic.cu: per-block shmem histogram, then one global
// atomicAdd per (block, expert).
__global__ void count_kernel(const int32_t* __restrict__ assignments,
                             int32_t*       __restrict__ expert_count,
                             int TK, int E) {
    extern __shared__ int32_t s_hist[];
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

constexpr int kWarpSize       = 32;
constexpr int kWarpsPerBlock  = 4;
constexpr int kVecPerLoad     = 8;       // 8 fp16 per uint4 = 16 bytes

// One warp per (token, k) pair. Lane 0 atomically reserves a slot within
// its expert; all lanes copy the embedding row in uint4 (16-byte) chunks.
__global__ void place_gather_kernel(const __half*  __restrict__ emb,
                                    const int32_t* __restrict__ asgn,
                                    const float*   __restrict__ wts,
                                    const int32_t* __restrict__ expert_start,
                                    int32_t*       __restrict__ cursor,
                                    __half*        __restrict__ packed,
                                    float*         __restrict__ packed_wts,
                                    int32_t*       __restrict__ perm,
                                    int K, int d, int TK) {
    int warp_in_block = threadIdx.x / kWarpSize;
    int lane          = threadIdx.x & (kWarpSize - 1);
    int idx           = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (idx >= TK) return;

    int t = idx / K;
    int e = asgn[idx];

    // Lane 0 reserves position, broadcasts to whole warp.
    int pos = 0;
    if (lane == 0) {
        int slot = atomicAdd(&cursor[e], 1);
        pos = expert_start[e] + slot;
    }
    pos = __shfl_sync(0xffffffff, pos, 0);

    const uint4* src_v = reinterpret_cast<const uint4*>(emb    + (size_t)t   * d);
    uint4*       dst_v = reinterpret_cast<uint4*>      (packed + (size_t)pos * d);
    const int d_v = d / kVecPerLoad;
    for (int i = lane; i < d_v; i += kWarpSize) dst_v[i] = src_v[i];

    if (lane == 0) {
        packed_wts[pos] = wts[idx];
        perm[pos]       = t;
    }
}

// Workspace: CUB scan temp, then per-expert atomic cursor.
constexpr std::size_t kAlign = 16;

struct CSortWorkspace {
    std::size_t cub_bytes;
    std::size_t total_bytes;
};

CSortWorkspace plan_workspace(const MoEParams& p) {
    CSortWorkspace ws{};
    cub::DeviceScan::ExclusiveSum<int32_t*, int32_t*>(
        nullptr, ws.cub_bytes,
        (int32_t*)nullptr, (int32_t*)nullptr, p.E);
    ws.cub_bytes  = (ws.cub_bytes + kAlign - 1) & ~(kAlign - 1);
    ws.total_bytes = ws.cub_bytes + (std::size_t)p.E * sizeof(int32_t);
    return ws;
}

class CSortScatter final : public IScatter {
public:
    const char* name() const override { return "csort"; }

    std::size_t workspace_bytes(const MoEParams& p) const override {
        return plan_workspace(p).total_bytes;
    }

    ScatterTimings run(const __half*  d_emb,
                       const int32_t* d_asgn,
                       const float*   d_wts,
                       MoEBuffers&    bufs,
                       const MoEParams& p,
                       cudaStream_t   stream) override {
        const int TK = p.T * p.K;

        if (p.d_model % kVecPerLoad != 0) {
            std::fprintf(stderr,
                "csort scatter requires d_model %% %d == 0 (got d=%d)\n",
                kVecPerLoad, p.d_model);
            std::exit(1);
        }

        CSortWorkspace ws = plan_workspace(p);
        void*    d_cub_tmp = bufs.strategy_ws;
        int32_t* d_cursor  = reinterpret_cast<int32_t*>(
            static_cast<char*>(bufs.strategy_ws) + ws.cub_bytes);

        ScatterTimings t;
        CudaTimer tc, tp, tg, tt;
        tt.start(stream);

        // 1. histogram
        device_zero_async(bufs.expert_count, (std::size_t)p.E, stream);
        tc.start(stream);
        {
            const int blk = 256, grd = (TK + blk - 1) / blk;
            count_kernel<<<grd, blk, p.E * sizeof(int32_t), stream>>>(
                d_asgn, bufs.expert_count, TK, p.E);
        }
        t.count_ms = tc.stop_ms(stream);

        // 2. exclusive prefix sum → expert_start
        tp.start(stream);
        cub::DeviceScan::ExclusiveSum(
            d_cub_tmp, ws.cub_bytes,
            bufs.expert_count, bufs.expert_start, p.E, stream);
        t.prefix_ms = tp.stop_ms(stream);

        // 3. place + gather (warp-per-row, vectorized)
        device_zero_async(d_cursor, (std::size_t)p.E, stream);
        tg.start(stream);
        {
            const int threads = kWarpSize * kWarpsPerBlock;     // 128
            const int grd     = (TK + kWarpsPerBlock - 1) / kWarpsPerBlock;
            place_gather_kernel<<<grd, threads, 0, stream>>>(
                d_emb, d_asgn, d_wts,
                bufs.expert_start, d_cursor,
                bufs.packed, bufs.packed_weights, bufs.perm,
                p.K, p.d_model, TK);
        }
        t.gather_ms = tg.stop_ms(stream);

        t.total_ms = tt.stop_ms(stream);
        return t;
    }
};

} // namespace

std::unique_ptr<IScatter> make_csort_scatter() {
    return std::make_unique<CSortScatter>();
}

} // namespace moe
