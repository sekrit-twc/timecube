#include <exception>
#include <iostream>
#include <string>
#include "cube.h"
#include "lut.h"

namespace {

void usage()
{
	std::cerr << "usage: test cubefile [x y z] [simd]\n";
}

} // namespace


int main(int argc, char **argv)
{
	if (argc < 2) {
		usage();
		return 1;
	}

	std::unique_ptr<timecube::Lut> lut;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	bool simd = false;

	try {
		if (argc >= 5) {
			x = std::stof(argv[2]);
			y = std::stof(argv[3]);
			z = std::stof(argv[4]);
		}

		if (argc >= 6)
			simd = !!std::stof(argv[5]);
	} catch (const std::exception &) {
		usage();
		return 1;
	}

	try {
		timecube::Cube cube = timecube::read_cube_from_file(argv[1]);
		std::cout << "first entry: " << cube.lut[0] << ' ' << cube.lut[1] << ' ' << cube.lut[2] << '\n';
		std::cout << "last entry: " << cube.lut[cube.lut.size() - 3] << ' ' << cube.lut[cube.lut.size() - 2] << ' ' << cube.lut[cube.lut.size() - 1] << '\n';

		if (!(lut = timecube::create_lut_impl(cube, simd)))
			throw std::runtime_error{ "failed to create LUT implementation" };
	} catch (const std::exception &e) {
		std::cerr << "failed to load CUBE: " << e.what() << '\n';
		return 1;
	}

	try {
		const void *src[3] = { &x, &y, &z };
		void *dst[3] = { &x, &y, &z };

		lut->process(src, dst, 1);
		std::cout << "result: (" << x << ", " << y << ", " << z << ")\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}

	return 0;
}
