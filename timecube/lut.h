#pragma once

#ifndef timecube_LUT_H_
#define timecube_LUT_H_

#include <memory>

namespace timecube {

struct Cube;

class Lut {
public:
	virtual void process(const void * const src[3], void * const dst[3], unsigned width) = 0;
};

std::unique_ptr<Lut> create_lut_impl(const Cube &cube, bool enable_simd);

} // namespace timecube

#endif // timecube_LUT_H_
