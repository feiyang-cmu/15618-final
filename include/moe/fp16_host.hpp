// include/moe/fp16_host.hpp
//
// Host-side fp16 <-> fp32 helpers plus deterministic random fill utilities
// used by bench_e2e to seed random expert weights. CUDA's __half2float /
// __float2half already work in host code under nvcc; these wrappers add
// array-level fill helpers on top.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cuda_fp16.h>

namespace moe {

inline float  h2f(std::uint16_t h) { __half v; std::memcpy(&v, &h, 2); return __half2float(v); }
inline std::uint16_t f2h(float f)  { __half v = __float2half(f); std::uint16_t h; std::memcpy(&h, &v, 2); return h; }

/// Fill buffer with values in [-half_range, +half_range] using a linear
/// congruential generator seeded from `seed`. Deterministic across runs.
inline void fill_random_half(std::uint16_t* buf, std::size_t n, std::uint32_t seed,
                             float half_range = 0.01f) {
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float x = ((float)(s >> 16) / 65536.0f - 0.5f) * 2.f * half_range;
        buf[i] = f2h(x);
    }
}

inline void fill_random_float(float* buf, std::size_t n, std::uint32_t seed,
                              float half_range = 0.05f) {
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = ((float)(s >> 16) / 65536.0f - 0.5f) * 2.f * half_range;
    }
}

} // namespace moe
