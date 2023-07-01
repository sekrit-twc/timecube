#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "cube.h"
#include "lut.h"
#include "x86/lut_x86.h"

namespace timecube {
namespace {

constexpr int32_t numeric_max(int bits) noexcept
{
	return (1L << bits) - 1;
}

constexpr int32_t integer_offset(const PixelFormat &format) noexcept
{
	return pixel_is_float(format.type) ? 0
		: !format.fullrange ? 16L << (format.depth - 8)
		: 0;
}

constexpr int32_t integer_range(const PixelFormat &format) noexcept
{
	return pixel_is_float(format.type) ? 1
		: format.fullrange ? numeric_max(format.depth)
		: 219L << (format.depth - 8);
}

inline std::pair<float, float> get_scale_offset(const PixelFormat &pixel_in, const PixelFormat &pixel_out)
{
	double range_in = integer_range(pixel_in);
	double offset_in = integer_offset(pixel_in);
	double range_out = integer_range(pixel_out);
	double offset_out = integer_offset(pixel_out);

	float scale = static_cast<float>(range_out / range_in);
	float offset = static_cast<float>(-offset_in * range_out / range_in + offset_out);

	return{ scale, offset };
}


struct Vector3 : public std::array<float, 3> {
	Vector3() = default;

	Vector3(float a, float b, float c) : std::array<float, 3>{ { a, b, c } } {}
};

Vector3 operator+(const Vector3 &lhs, const Vector3 &rhs)
{
	return{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2] };
}

Vector3 operator-(const Vector3 &lhs, const Vector3 &rhs)
{
	return{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2] };
}

Vector3 operator*(float lhs, const Vector3 &rhs)
{
	return{ lhs * rhs[0], lhs * rhs[1], lhs * rhs[2] };
}

Vector3 operator*(const Vector3 &lhs, float rhs)
{
	return rhs * lhs;
}

template <class T>
T interp(T lo, T hi, float dist)
{
	return (1.0f - dist) * lo + dist * hi;
}

Vector3 trilinear_interp(float r, float g, float b, const Vector3 *lut, uint_least32_t dim)
{
	Vector3 vert000 = lut[0 + 0 * dim + 0 * dim * dim];
	Vector3 vert001 = lut[1 + 0 * dim + 0 * dim * dim];
	Vector3 vert010 = lut[0 + 1 * dim + 0 * dim * dim];
	Vector3 vert011 = lut[1 + 1 * dim + 0 * dim * dim];
	Vector3 vert100 = lut[0 + 0 * dim + 1 * dim * dim];
	Vector3 vert101 = lut[1 + 0 * dim + 1 * dim * dim];
	Vector3 vert110 = lut[0 + 1 * dim + 1 * dim * dim];
	Vector3 vert111 = lut[1 + 1 * dim + 1 * dim * dim];

	Vector3 tmp0 = interp(vert000, vert001, r);
	Vector3 tmp1 = interp(vert100, vert101, r);
	Vector3 tmp2 = interp(vert010, vert011, r);
	Vector3 tmp3 = interp(vert110, vert111, r);

	tmp0 = interp(tmp0, tmp2, g);
	tmp1 = interp(tmp1, tmp3, g);

	tmp0 = interp(tmp0, tmp1, b);
	return tmp0;
}

template <class T>
void minmax(T &x, T &y)
{
	T minval = std::min(x, y);
	T maxval = std::max(x, y);
	x = minval;
	y = maxval;
}

template <class T>
void sort3(T &x, T &y, T &z)
{
	minmax(x, z);
	minmax(x, y);
	minmax(y, z);
}

Vector3 tetrahedral_interp(float r, float g, float b, const Vector3 *lut, uint_least32_t dim)
{
	float x = r, y = g, z = b;
	uint_least32_t diag = 1 + dim + dim * dim;
	uint_least32_t disp1, disp2;

	Vector3 vert[4];
	float w[4];

	sort3(x, y, z);
	disp1 = z == r ? 1 : z == g ? dim : dim * dim;
	disp2 = x == r ? 1 : x == g ? dim : dim * dim;

	vert[0] = lut[0];
	vert[1] = lut[diag];
	vert[2] = lut[disp1];
	vert[3] = lut[diag - disp2];

	w[0] = 1.0f - z;
	w[1] = x;
	w[2] = z - y;
	w[3] = y - x;

	return w[0] * vert[0] + w[1] * vert[1] + w[2] * vert[2] + w[3] * vert[3];
}


class Lut1DFilter_C : public Lut1DFilter {
	std::vector<float> m_lut;
	float m_scale;
	float m_offset;
public:
	Lut1DFilter_C(const Cube &cube, unsigned width, unsigned height, unsigned plane) :
		Lut1DFilter(width, height),
		m_scale{},
		m_offset{}
	{
		m_scale = 1.0f / (cube.domain_max[plane] - cube.domain_min[plane]);
		m_offset = cube.domain_min[plane] * m_scale;

		for (size_t i = 0; i < cube.n; ++i) {
			m_lut[i] = cube.lut[i * 3 + plane];
		}
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
                 unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		uint_least32_t lut_max = static_cast<uint_least32_t>(m_lut.size() - 1);
		float lut_clamp = std::nextafterf(static_cast<float>(lut_max), -INFINITY);

		const float *src_p = in->get_line<float>(i);
		float *dst_p = out->get_line<float>(i);

		for (unsigned i = left; i < right; ++i) {
			float x, d;
			uint_least32_t idx;

			x = src_p[i];
			x = (x * m_scale + m_offset) * lut_max;

			x = std::min(std::max(x, 0.0f), lut_clamp);
			idx = static_cast<uint_least32_t>(x);
			d = x - idx;

			x = interp(m_lut[idx], m_lut[idx + 1], d);
			dst_p[i] = x;
		}
	}
};

class Lut3DFilter_C : public Lut3DFilter {
protected:
	std::vector<Vector3> m_lut;
	uint_least32_t m_dim;
	float m_scale[3];
	float m_offset[3];

	Lut3DFilter_C(const Cube &cube, unsigned width, unsigned height) :
		Lut3DFilter(width, height),
		m_dim{ cube.n },
		m_scale{},
		m_offset{}
	{
		for (unsigned i = 0; i < 3; ++i) {
			m_scale[i] = 1.0f / (cube.domain_max[i] - cube.domain_min[i]);
			m_offset[i] = cube.domain_min[i] * m_scale[i];
		}

		m_lut.resize(m_dim * m_dim * m_dim);

		for (size_t i = 0; i < m_lut.size(); ++i) {
			m_lut[i][0] = cube.lut[i * 3 + 0];
			m_lut[i][1] = cube.lut[i * 3 + 1];
			m_lut[i][2] = cube.lut[i * 3 + 2];
		}
	}
};

class TrilinearFilter_C : public Lut3DFilter_C {
public:
	TrilinearFilter_C(const Cube &cube, unsigned width, unsigned height) : Lut3DFilter_C(cube, width, height) {}

	void process(const graphengine::BufferDescriptor in[], const graphengine::BufferDescriptor out[],
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *src_r = in[0].get_line<float>(i);
		const float *src_g = in[1].get_line<float>(i);
		const float *src_b = in[2].get_line<float>(i);
		float *dst_r = out[0].get_line<float>(i);
		float *dst_g = out[1].get_line<float>(i);
		float *dst_b = out[2].get_line<float>(i);

		uint_least32_t lut_max = m_dim - 1;
		float lut_clamp = std::nextafter(static_cast<float>(lut_max), -INFINITY);

		for (unsigned i = left; i < right; ++i) {
			float r, g, b;
			float dist_r, dist_g, dist_b;
			uint_least32_t idx_r, idx_g, idx_b;
			uint_least32_t idx;

			Vector3 interp_result;

			r = src_r[i];
			g = src_g[i];
			b = src_b[i];

			r = (r * m_scale[0] + m_offset[0]) * lut_max;
			g = (g * m_scale[1] + m_offset[1]) * lut_max;
			b = (b * m_scale[2] + m_offset[2]) * lut_max;

			r = std::min(std::max(r, 0.0f), lut_clamp);
			g = std::min(std::max(g, 0.0f), lut_clamp);
			b = std::min(std::max(b, 0.0f), lut_clamp);

			idx_r = static_cast<uint_least32_t>(r);
			idx_g = static_cast<uint_least32_t>(g);
			idx_b = static_cast<uint_least32_t>(b);
			idx = idx_r + idx_g * m_dim + idx_b * m_dim * m_dim;

			dist_r = r - idx_r;
			dist_g = g - idx_g;
			dist_b = b - idx_b;

			interp_result = trilinear_interp(dist_r, dist_g, dist_b, m_lut.data() + idx, m_dim);
			r = interp_result[0];
			g = interp_result[1];
			b = interp_result[2];

			dst_r[i] = r;
			dst_g[i] = g;
			dst_b[i] = b;
		}
	}
};

class TetrahedralFilters_C : public Lut3DFilter_C {
public:
	TetrahedralFilters_C(const Cube &cube, unsigned width, unsigned height) : Lut3DFilter_C(cube, width, height) {}

	void process(const graphengine::BufferDescriptor in[], const graphengine::BufferDescriptor out[],
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *src_r = in[0].get_line<float>(i);
		const float *src_g = in[1].get_line<float>(i);
		const float *src_b = in[2].get_line<float>(i);
		float *dst_r = out[0].get_line<float>(i);
		float *dst_g = out[1].get_line<float>(i);
		float *dst_b = out[2].get_line<float>(i);

		uint_least32_t lut_max = m_dim - 1;
		float lut_clamp = std::nextafter(static_cast<float>(lut_max), -INFINITY);

		for (unsigned i = left; i < right; ++i) {
			float r, g, b;
			float dist_r, dist_g, dist_b;
			uint_least32_t idx_r, idx_g, idx_b;
			uint_least32_t idx;

			Vector3 interp_result;

			r = src_r[i];
			g = src_g[i];
			b = src_b[i];

			r = (r * m_scale[0] + m_offset[0]) * lut_max;
			g = (g * m_scale[1] + m_offset[1]) * lut_max;
			b = (b * m_scale[2] + m_offset[2]) * lut_max;

			r = std::min(std::max(r, 0.0f), lut_clamp);
			g = std::min(std::max(g, 0.0f), lut_clamp);
			b = std::min(std::max(b, 0.0f), lut_clamp);

			idx_r = static_cast<uint_least32_t>(r);
			idx_g = static_cast<uint_least32_t>(g);
			idx_b = static_cast<uint_least32_t>(b);
			idx = idx_r + idx_g * m_dim + idx_b * m_dim * m_dim;

			dist_r = r - idx_r;
			dist_g = g - idx_g;
			dist_b = b - idx_b;

			interp_result = tetrahedral_interp(dist_r, dist_g, dist_b, m_lut.data() + idx, m_dim);

			dst_r[i] = interp_result[0];
			dst_g[i] = interp_result[1];
			dst_b[i] = interp_result[2];
		}
	}
};


template <class T>
void to_float(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned)
{
	const T *srcp = static_cast<const T *>(src);
	float *dstp = static_cast<float *>(dst);

	for (unsigned i = left; i < right; ++i) {
		dstp[i] = (static_cast<float>(srcp[i]) * scale) + offset;
	}
}

template <class T>
void from_float(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned depth)
{
	const float *srcp = static_cast<const float *>(src);
	T *dstp = static_cast<T *>(dst);
	long maxval = (1UL << depth) - 1;

	for (unsigned i = left; i < right; ++i) {
		dstp[i] = static_cast<T>(std::min(std::lrint(srcp[i] * scale + offset), maxval));
	}
}

pixel_io_func select_from_float_func(PixelType to)
{
	switch (to) {
	case PixelType::BYTE:
		return from_float<uint8_t>;
	case PixelType::WORD:
		return from_float<uint16_t>;
	default:
		return nullptr;
	}
}

pixel_io_func select_to_float_func(PixelType from)
{
	switch (from) {
	case PixelType::BYTE:
		return to_float<uint8_t>;
	case PixelType::WORD:
		return to_float<uint16_t>;
	default:
		return nullptr;
	}
}

} // namespace


PixelIOFilter::PixelIOFilter(from_float_tag, unsigned width, unsigned height, const PixelFormat &to, pixel_io_func func) :
	m_desc{},
	m_func{ func },
	m_scale{},
	m_offset{},
	m_depth{ to.depth }
{
	m_desc.format = { width, height, pixel_size(to.type) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;
	m_desc.alignment_mask = 0x15;

	std::tie(m_scale, m_offset) = get_scale_offset({ PixelType::FLOAT }, to);
}

PixelIOFilter::PixelIOFilter(to_float_tag, unsigned width, unsigned height, const PixelFormat &from, pixel_io_func func) :
	m_desc{},
	m_func{ func },
	m_scale{},
	m_offset{},
	m_depth{ from.depth }
{
	m_desc.format = { width, height, pixel_size(PixelType::FLOAT) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;
	m_desc.alignment_mask = 0x15;

	std::tie(m_scale, m_offset) = get_scale_offset(from, { PixelType::FLOAT });
}

void PixelIOFilter::process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
                            unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	m_func(in->get_line(i), out->get_line(i), left, right, m_scale, m_offset, m_depth);
}

Lut1DFilter::Lut1DFilter(unsigned width, unsigned height) : m_desc{}
{
	m_desc.format = { width, height, sizeof(float) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;
	m_desc.flags.in_place = 1;
}

Lut3DFilter::Lut3DFilter(unsigned width, unsigned height) : m_desc{}
{
	m_desc.format = { width, height, sizeof(float) };
	m_desc.num_deps = 3;
	m_desc.num_planes = 3;
	m_desc.step = 1;
	m_desc.flags.in_place = 1;
}


std::unique_ptr<graphengine::Filter> create_to_float_impl(unsigned width, unsigned height, const PixelFormat &from, int simd)
{
	pixel_io_func func = nullptr;
#ifdef CUBE_X86
	func = select_to_float_func_x86(from.type, simd);
#endif
	if (!func)
		func = select_to_float_func(from.type);

	if (!func)
		throw std::invalid_argument{ "unsupported pixel type" };

	return std::make_unique<PixelIOFilter>(PixelIOFilter::TO_FLOAT, width, height, from, func);
}

std::unique_ptr<graphengine::Filter> create_from_float_impl(unsigned width, unsigned height, const PixelFormat &to, int simd)
{
	pixel_io_func func = nullptr;
#ifdef CUBE_X86
	func = select_from_float_func_x86(to.type, simd);
#endif
	if (!func)
		func = select_from_float_func(to.type);

	if (!func)
		throw std::invalid_argument{ "unsupported pixel type" };

	return std::make_unique<PixelIOFilter>(PixelIOFilter::FROM_FLOAT, width, height, to, func);
}

std::unique_ptr<graphengine::Filter> create_lut1d_impl(const Cube &cube, unsigned width, unsigned height, unsigned plane, Interpolation, int)
{
	if (cube.is_3d)
		throw std::invalid_argument{ "wrong LUT type" };

	return std::make_unique<Lut1DFilter_C>(cube, width, height, plane);
}

std::unique_ptr<graphengine::Filter> create_lut3d_impl(const Cube &cube, unsigned width, unsigned height, Interpolation interp, int simd)
{
	if (!cube.is_3d)
		throw std::invalid_argument{ "wrong LUT type" };

	std::unique_ptr<graphengine::Filter> ret;
#ifdef CUBE_X86
	ret = create_lut3d_impl_x86(cube, width, height, interp, simd);
#endif
	if (!ret) {
		if (interp == Interpolation::TETRA)
			ret = std::make_unique<TetrahedralFilters_C>(cube, width, height);
		else
			ret = std::make_unique<TrilinearFilter_C>(cube, width, height);
	}

	return ret;
}

} // namespace timecube
