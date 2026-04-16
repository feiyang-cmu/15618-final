// src/common/npy_io.cpp

#include "moe/npy_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace moe {

std::size_t NpyArray::elt_bytes() const {
    if (dtype == "<f2" || dtype == "<u2" || dtype == "<i2") return 2;
    if (dtype == "<f4" || dtype == "<i4" || dtype == "<u4") return 4;
    return 0;
}

namespace {

[[noreturn]] void die(const char* msg, const std::string& path) {
    std::fprintf(stderr, "[npy] %s: %s\n", msg, path.c_str());
    std::exit(1);
}

void must_read(void* dst, std::size_t n, std::FILE* fp, const std::string& path) {
    if (std::fread(dst, 1, n, fp) != n) die("short read", path);
}

} // namespace

NpyArray load_npy(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) die("cannot open", path);

    char magic[6];
    must_read(magic, 6, fp, path);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) die("bad magic", path);

    std::uint8_t major = 0, minor = 0;
    must_read(&major, 1, fp, path);
    must_read(&minor, 1, fp, path);

    std::size_t header_len = 0;
    if (major == 1) {
        std::uint16_t h = 0;
        must_read(&h, 2, fp, path);
        header_len = h;
    } else {
        std::uint32_t h = 0;
        must_read(&h, 4, fp, path);
        header_len = h;
    }

    std::string header(header_len, ' ');
    must_read(header.data(), header_len, fp, path);

    NpyArray arr;

    // Parse 'descr': '<..'
    auto d_pos = header.find("'descr':");
    if (d_pos == std::string::npos) die("no descr", path);
    auto q1 = header.find('\'', d_pos + 8);
    auto q2 = header.find('\'', q1 + 1);
    arr.dtype = header.substr(q1 + 1, q2 - q1 - 1);

    // Parse 'shape': (R, C)  — 2-D only
    auto s_pos = header.find("'shape':");
    if (s_pos == std::string::npos) die("no shape", path);
    auto lp = header.find('(', s_pos);
    auto rp = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    auto comma = shape_str.find(',');
    if (comma == std::string::npos) die("non-2D shape", path);
    arr.rows = std::strtoul(shape_str.substr(0, comma).c_str(), nullptr, 10);
    arr.cols = std::strtoul(shape_str.substr(comma + 1).c_str(), nullptr, 10);

    std::size_t elt = arr.elt_bytes();
    if (elt == 0) {
        std::fprintf(stderr, "[npy] unsupported dtype '%s' in %s\n",
                     arr.dtype.c_str(), path.c_str());
        std::exit(1);
    }

    std::size_t nbytes = arr.rows * arr.cols * elt;
    arr.data.resize(nbytes);
    must_read(arr.data.data(), nbytes, fp, path);
    std::fclose(fp);

    std::printf("[npy] %-56s shape=(%zu,%zu) dtype=%s\n",
                path.c_str(), arr.rows, arr.cols, arr.dtype.c_str());
    return arr;
}

} // namespace moe
