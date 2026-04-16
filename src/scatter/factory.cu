// src/scatter/factory.cpp
//
// IScatter factory. One dispatch point that knows every registered strategy.
// Strategy TUs expose `make_<name>_scatter()` free functions (declared in
// include/moe/scatter.cuh); this file just switches on the string name.

#include "moe/scatter.cuh"

namespace moe {

std::unique_ptr<IScatter> make_scatter(const std::string& name) {
    if (name == "atomic") return make_atomic_scatter();
    if (name == "sort")   return make_sort_scatter();
    if (name == "warp")   return make_warp_scatter();
    return nullptr;
}

} // namespace moe
