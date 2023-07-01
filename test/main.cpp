#include <exception>
#include <iostream>
#include <string>
#include <graphengine/filter.h>
#include "cube.h"
#include "lut.h"

namespace {

void usage()
{
	std::cerr << "usage: test cubefile [x y z] [interp] [simd]\n";
}

} // namespace


int main(int argc, char **argv)
{
	if (argc < 2) {
		usage();
		return 1;
	}

	std::unique_ptr<graphengine::Filter> lut[3];
	alignas(64) float x = 0.0f;
	alignas(64) float y = 0.0f;
	alignas(64) float z = 0.0f;
	timecube::Interpolation interp = timecube::Interpolation::LINEAR;
	int simd = 0;

	try {
		if (argc >= 5) {
			x = std::stof(argv[2]);
			y = std::stof(argv[3]);
			z = std::stof(argv[4]);
		}

		if (argc >= 6)
			interp = static_cast<timecube::Interpolation>(std::stoi(argv[5]));

		if (argc >= 7)
			simd = std::stoi(argv[6]);
	} catch (const std::exception &) {
		usage();
		return 1;
	}

	try {
		timecube::Cube cube = timecube::read_cube_from_file(argv[1]);
		std::cout << "first entry: " << cube.lut[0] << ' ' << cube.lut[1] << ' ' << cube.lut[2] << '\n';
		std::cout << "last entry: " << cube.lut[cube.lut.size() - 3] << ' ' << cube.lut[cube.lut.size() - 2] << ' ' << cube.lut[cube.lut.size() - 1] << '\n';

		if (cube.is_3d) {
			if (!(lut[0] = timecube::create_lut3d_impl(cube, 1, 1, interp, simd)))
				throw std::runtime_error{ "failed to create LUT implementation" };
		} else {
			if (!(lut[0] = timecube::create_lut1d_impl(cube, 1, 1, 0, interp, simd)))
				throw std::runtime_error{ "failed to create LUT implementation" };
			if (!(lut[1] = timecube::create_lut1d_impl(cube, 1, 1, 1, interp, simd)))
				throw std::runtime_error{ "failed to create LUT implementation" };
			if (!(lut[2] = timecube::create_lut1d_impl(cube, 1, 1, 2, interp, simd)))
				throw std::runtime_error{ "failed to create LUT implementation" };
		}
	} catch (const std::exception &e) {
		std::cerr << "failed to load CUBE: " << e.what() << '\n';
		return 1;
	}

	try {
		graphengine::BufferDescriptor buffer[3] = {
			{ &x, 64, graphengine::BUFFER_MAX },
			{ &y, 64, graphengine::BUFFER_MAX },
			{ &z, 64, graphengine::BUFFER_MAX },
		};

		if (!lut[1] /* is_3d */) {
			lut[0]->process(buffer, buffer, 0, 0, 1, nullptr, nullptr);
		} else {
			lut[0]->process(&buffer[0], &buffer[0], 0, 0, 1, nullptr, nullptr);
			lut[1]->process(&buffer[1], &buffer[1], 0, 0, 1, nullptr, nullptr);
			lut[2]->process(&buffer[2], &buffer[2], 0, 0, 1, nullptr, nullptr);
		}
		std::cout << "result: (" << x << ", " << y << ", " << z << ")\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}

	return 0;
}
