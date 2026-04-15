// include/moe/npy_io.hpp
//
// Minimal NumPy .npy loader. Supports 2-D arrays with dtype in
// { "<f2" (fp16), "<f4" (fp32), "<i4" (int32) } on little-endian hosts.
// Sufficient for the files produced by data/gen_synthetic_routing.py and
// data/collect_qwen_routing.py.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe {

struct NpyArray {
    std::vector<std::uint8_t> data;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::string dtype;          ///< "<f2" | "<f4" | "<i4"

    std::size_t elt_bytes() const;

    /// Typed views — caller is responsible for matching dtype.
    const std::uint16_t* as_fp16() const { return reinterpret_cast<const std::uint16_t*>(data.data()); }
    const float*         as_fp32() const { return reinterpret_cast<const float*>(data.data()); }
    const std::int32_t*  as_i32()  const { return reinterpret_cast<const std::int32_t*>(data.data()); }
};

/// Load and validate. Aborts on IO / dtype errors.
NpyArray load_npy(const std::string& path);

} // namespace moe
