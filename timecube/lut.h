#pragma once

#ifndef TIMECUBE_LUT_H_
#define TIMECUBE_LUT_H_

#include <memory>
#include <graphengine/filter.h>

namespace timecube {

struct Cube;


enum class PixelType {
	BYTE,
	WORD,
	HALF,
	FLOAT,
};

struct PixelFormat {
	PixelType type;
	unsigned depth;
	bool fullrange;
};

constexpr unsigned pixel_size(PixelType type)
{
	switch (type) {
	case PixelType::BYTE:
		return sizeof(uint8_t);
	case PixelType::WORD:
	case PixelType::HALF:
		return sizeof(uint16_t);
	case PixelType::FLOAT:
		return sizeof(float);
	default:
		throw 1;
	}
}

constexpr bool pixel_is_float(PixelType type) { return type == PixelType::HALF || type == PixelType::FLOAT; }


typedef void (*pixel_io_func)(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned depth);

class PixelIOFilter : public graphengine::Filter {
public:
	struct from_float_tag {};
	struct to_float_tag {};

	static constexpr from_float_tag FROM_FLOAT{};
	static constexpr to_float_tag TO_FLOAT{};
protected:
	graphengine::FilterDescriptor m_desc;
	pixel_io_func m_func;
	float m_scale;
	float m_offset;
	unsigned m_depth;
public:
	PixelIOFilter(from_float_tag, unsigned width, unsigned height, const PixelFormat &to, pixel_io_func func);

	PixelIOFilter(to_float_tag, unsigned width, unsigned height, const PixelFormat &from, pixel_io_func func);

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
                 unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

class Lut1DFilter : public graphengine::Filter {
protected:
	graphengine::FilterDescriptor m_desc;
public:
	Lut1DFilter(unsigned width, unsigned height);

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}
};

class Lut3DFilter : public graphengine::Filter {
protected:
	graphengine::FilterDescriptor m_desc;
public:
	Lut3DFilter(unsigned width, unsigned height);

	int version() const noexcept override { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}
};

std::unique_ptr<graphengine::Filter> create_to_float_impl(unsigned width, unsigned height, const PixelFormat &from, int simd);

std::unique_ptr<graphengine::Filter> create_from_float_impl(unsigned width, unsigned height, const PixelFormat &to, int simd);

std::unique_ptr<graphengine::Filter> create_lut1d_impl(const Cube &cube, unsigned width, unsigned height, unsigned plane, int simd);

std::unique_ptr<graphengine::Filter> create_lut3d_impl(const Cube &cube, unsigned width, unsigned height, int simd);

} // namespace timecube

#endif // TIMECUBE_LUT_H_
