#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include "cube.h"
#include "lut.h"

int main(int argc, char **argv)
{
	unsigned niter = 4000000;
	bool simd = true;

	try {
		if (argc >= 2)
			niter = std::stoi(argv[1]);
		if (argc >= 3)
			simd = !!std::stoi(argv[2]);
	} catch (const std::exception &) {
		std::cerr << "usage: benchmark [niter] [simd]\n";
		return 1;
	}

	timecube::Cube cube;
	cube.lut = {
		0.0f, 0.0f, 0.0f, // R0 G0 B0
		1.0f, 0.0f, 0.0f, // R1 G0 B0
		0.0f, 1.0f, 0.0f, // R0 G1 B0
		1.0f, 1.0f, 0.0f, // R1 G1 B0
		0.0f, 0.0f, 1.0f, // R0 G0 B1
		1.0f, 0.0f, 1.0f, // R1 G0 B1
		0.0f, 1.0f, 1.0f, // R0 G1 B1
		1.0f, 1.0f, 1.0f, // R1 G1 B1
	};
	cube.is_3d = true;
	cube.n = 2;

	std::unique_ptr<timecube::Lut> lut;

	try {
		if (!(lut = timecube::create_lut_impl(cube, simd)))
			throw std::runtime_error{ "failed to create LUT implementation" };
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
	}

	alignas(32) float r[1024] = { 0 };
	alignas(32) float g[1024] = { 0 };
	alignas(32) float b[1024] = { 0 };

	const float *src[3] = { r, g, b };
	float *dst[3] = { r, g, b };

	auto start = std::chrono::high_resolution_clock::now();
	for (unsigned i = 0; i < niter; ++i) {
		lut->process(src, dst, sizeof(r) / sizeof(r[0]));
	}
	auto end = std::chrono::high_resolution_clock::now();

	double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
	double pels_per_s = (sizeof(r) / sizeof(r[0])) * niter / duration;
	double ns_per_pel = 1e9 / pels_per_s;
	std::cout << "elapsed: " << duration << " (" << ns_per_pel << " ns/pel)\n";

	if (std::getenv("INTERACTIVE")) {
		std::cout << "press any key to continue...\n";
		std::fgetc(stdin);
	}

	return 0;
}
