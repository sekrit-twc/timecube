#include <chrono>
#include <climits>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <graphengine/filter.h>
#include "cube.h"
#include "lut.h"

int main(int argc, char **argv)
{
	unsigned niter = 4000000;
	timecube::Interpolation interp = timecube::Interpolation::LINEAR;
	int simd = INT_MAX;

	try {
		if (argc >= 2)
			niter = std::stoi(argv[1]);
		if (argc >= 3)
			interp = static_cast<timecube::Interpolation>(std::stoi(argv[2]));
		if (argc >= 4)
			simd = std::stoi(argv[3]);
	} catch (const std::exception &) {
		std::cerr << "usage: benchmark [niter] [interp] [simd]\n";
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

	std::unique_ptr<graphengine::Filter> lut;
	alignas(64) float r[1024] = { 0 };
	alignas(64) float g[1024] = { 0 };
	alignas(64) float b[1024] = { 0 };

	try {
		if (!(lut = timecube::create_lut3d_impl(cube, sizeof(r) / sizeof(r[0]), 1, interp, simd)))
			throw std::runtime_error{ "failed to create LUT implementation" };
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
	}

	graphengine::BufferDescriptor buffers[3] = {
		{ r, sizeof(r), graphengine::BUFFER_MAX },
		{ g, sizeof(g), graphengine::BUFFER_MAX },
		{ b, sizeof(b), graphengine::BUFFER_MAX },
	};

	auto start = std::chrono::high_resolution_clock::now();
	for (unsigned i = 0; i < niter; ++i) {
		lut->process(buffers, buffers, 0, 0, sizeof(r) / sizeof(r[0]), nullptr, nullptr);
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
