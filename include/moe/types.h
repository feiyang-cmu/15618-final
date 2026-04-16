// include/moe/types.h
//
// Plain-data types shared across the MoE token-packing library.
//
// Design note: every pointer in MoEBuffers is a raw device pointer owned by
// the caller. This keeps the surface trivially interoperable with
// torch::Tensor::data_ptr<T>() for a later Pybind11 wrapper.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace moe {

/// Static problem dimensions for one MoE layer forward pass.
struct MoEParams {
    int T;        ///< number of tokens
    int K;        ///< top-K experts per token
    int E;        ///< total number of experts
    int d_model;  ///< hidden dimension
    int d_ffn;    ///< FFN inner dimension (used by expert GEMMs)
};

/// Device-side buffers produced by a scatter strategy and consumed by the
/// expert GEMMs / unscatter stage. All pointers are caller-owned.
struct MoEBuffers {
    // Scatter outputs.
    __half*  packed;          ///< [T*K, d_model]   packed token rows
    float*   packed_weights;  ///< [T*K]            per-slot routing weight
    int32_t* perm;            ///< [T*K]            perm[pos] = source token_idx
    int32_t* expert_count;    ///< [E]              tokens assigned per expert
    int32_t* expert_start;    ///< [E+1]            exclusive prefix sum of counts

    // Strategy-owned scratch (allocated by the app using workspace_bytes()).
    void*  strategy_ws;
    size_t strategy_ws_bytes;
};

/// Input distribution tag for benchmark harnesses.
enum class Distribution { Uniform, Zipf };

} // namespace moe
