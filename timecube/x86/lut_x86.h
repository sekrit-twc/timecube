#pragma once

#ifdef CUBE_X86

#ifndef TIMECUBE_LUT_X86_H_
#define TIMECUBE_LUT_X86_H_

#include <memory>
#include "lut.h"

namespace graphengine {
class Filter;
}

namespace timecube {

struct Cube;

#define DEFINE_IO_FUNC(from, to, isa) \
  void from##_to_##to##_##isa(const void *, void *, unsigned, unsigned, float, float, unsigned);

#define DEFINE_IO_FUNC_ALL(from, to) \
  DEFINE_IO_FUNC(from, to, sse41) \
  DEFINE_IO_FUNC(from, to, avx2) \
  DEFINE_IO_FUNC(from, to, avx512)

DEFINE_IO_FUNC_ALL(byte, float)
DEFINE_IO_FUNC_ALL(word, float)
DEFINE_IO_FUNC_ALL(float, byte)
DEFINE_IO_FUNC_ALL(float, word)

DEFINE_IO_FUNC(half, float, avx2)
DEFINE_IO_FUNC(half, float, avx512)
DEFINE_IO_FUNC(float, half, avx2)
DEFINE_IO_FUNC(float, half, avx512)

std::unique_ptr<graphengine::Filter> create_lut3d_impl_sse41(const Cube &cube, unsigned width, unsigned height);
std::unique_ptr<graphengine::Filter> create_lut3d_impl_avx2(const Cube &cube, unsigned width, unsigned height);
std::unique_ptr<graphengine::Filter> create_lut3d_impl_avx512(const Cube &cube, unsigned width, unsigned height);

std::unique_ptr<graphengine::Filter> create_lut3d_impl_x86(const Cube &cube, unsigned width, unsigned height, int simd);

pixel_io_func select_from_float_func_x86(PixelType from, int simd);
pixel_io_func select_to_float_func_x86(PixelType to, int simd);

} // namespace timecube
#endif // TIMECUBE_LUT_X86_H_

#endif // CUBE_X86
