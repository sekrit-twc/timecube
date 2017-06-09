#include <exception>
#include <iostream>
#include <string>
#include "cube.h"
#include "lut.h"

int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cerr << "usage: test cubefile [x y z]\n";
		return 1;
	}

	timecube::Cube cube;

	try {
		cube = timecube::read_cube_from_file(argv[1]);
		std::cout << "first entry: " << cube.lut[0] << ' ' << cube.lut[1] << ' ' << cube.lut[2] << '\n';
		std::cout << "last entry: " << cube.lut[cube.lut.size() - 3] << ' ' << cube.lut[cube.lut.size() - 2] << ' ' << cube.lut[cube.lut.size() - 1] << '\n';
	} catch (const std::exception &e) {
		std::cerr << "failed to open CUBE: " << e.what() << '\n';
		return 1;
	}

	if (argc < 5)
		return 0;

	try {
		std::unique_ptr<timecube::Lut> lut = timecube::create_lut_impl(cube, false);
		float x, y, z;

		if (!lut)
			throw std::runtime_error{ "failed to create LUT implementation" };

		try {
			x = std::stof(argv[2]);
			y = std::stof(argv[3]);
			z = std::stof(argv[4]);
		} catch (const std::exception &) {
			std::cerr << "expected 3 numbers\n";
			return 1;
		}

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
