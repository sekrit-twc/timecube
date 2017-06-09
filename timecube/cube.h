#pragma once

#ifndef timecube_CUBE_H_
#define timecube_CUBE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace timecube {

struct Cube {
	std::string title;
	std::vector<float> lut; // B in outer-most dimension, R in inner-most dimension.
	uint_least32_t n;
	float domain_min[3];
	float domain_max[3];
	bool is_3d;

	Cube() : n{}, domain_min{}, domain_max{ 1.0f, 1.0f, 1.0f }, is_3d{} {}
};

Cube read_cube_from_file(const char *path);

} // namespace timecube

#endif /* timecube_CUBE_H_ */
