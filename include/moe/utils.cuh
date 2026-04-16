// include/moe/utils.cuh
//
// Minimal CUDA host-side utilities: error check macro, event-based timer,
// typed device allocator. Header-only to keep translation units simple.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace moe {

#define MOE_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(_err));        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

/// RAII wrapper around a pair of cudaEvents. Measures elapsed ms on one stream.
class CudaTimer {
public:
    CudaTimer() {
        MOE_CUDA_CHECK(cudaEventCreate(&start_));
        MOE_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaTimer(const CudaTimer&)            = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t s = 0) { MOE_CUDA_CHECK(cudaEventRecord(start_, s)); }

    /// Records stop, synchronizes, returns elapsed ms.
    float stop_ms(cudaStream_t s = 0) {
        MOE_CUDA_CHECK(cudaEventRecord(stop_, s));
        MOE_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        MOE_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

/// Allocate n elements of T on the current device. Caller frees with cudaFree.
template <class T>
inline T* device_alloc(std::size_t n) {
    T* p = nullptr;
    MOE_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

/// Convenience: zero n elements of T on a stream.
template <class T>
inline void device_zero_async(T* p, std::size_t n, cudaStream_t s = 0) {
    MOE_CUDA_CHECK(cudaMemsetAsync(p, 0, n * sizeof(T), s));
}

} // namespace moe
