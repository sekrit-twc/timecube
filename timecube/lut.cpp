#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>
#include "cube.h"
#include "lut.h"

namespace timecube {
namespace {

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

Vector3 trilinear_interp(const Vector3 tri[2][2][2], float dist_x, float dist_y, float dist_z)
{
	Vector3 tmp0 = interp(tri[0][0][0], tri[1][0][0], dist_x);
	Vector3 tmp1 = interp(tri[0][0][1], tri[1][0][1], dist_x);
	Vector3 tmp2 = interp(tri[0][1][0], tri[1][1][0], dist_x);
	Vector3 tmp3 = interp(tri[0][1][1], tri[1][1][1], dist_x);

	tmp0 = interp(tmp0, tmp2, dist_y);
	tmp1 = interp(tmp1, tmp3, dist_y);

	tmp0 = interp(tmp0, tmp1, dist_z);
	return tmp0;
}

class Lut1D : public Lut {
	std::vector<float> m_lut[3];
	float m_scale[3];
	float m_offset[3];
public:
	explicit Lut1D(const Cube &cube) :
		m_scale{},
		m_offset{}
	{
		for (unsigned i = 0; i < 3; ++i) {
			m_lut[i].resize(cube.n);
			m_scale[i] = 1.0f / (cube.domain_max[i] - cube.domain_min[i]);
			m_offset[i] = cube.domain_min[i] * m_scale[i];
		}

		for (size_t i = 0; i < cube.n; ++i) {
			m_lut[0][i] = cube.lut[i * 3 + 0];
			m_lut[1][i] = cube.lut[i * 3 + 1];
			m_lut[2][i] = cube.lut[i * 3 + 2];
		}
	}

	void process(const void * const src[3], void * const dst[3], unsigned width) override
	{
		unsigned lut_max = static_cast<unsigned>(m_lut[0].size() - 1);

		for (unsigned p = 0; p < 3; ++p) {
			const float *src_p = static_cast<const float *>(src[p]);
			float *dst_p = static_cast<float *>(dst[p]);

			for (unsigned i = 0; i < width; ++i) {
				float x, d;
				unsigned idx;

				x = src_p[i];
				x = (x * m_scale[p] + m_offset[p]) * lut_max;

				idx = std::min(static_cast<unsigned>(x), lut_max);
				d = x - idx;

				x = interp(m_lut[p][idx], m_lut[p][idx + 1], d);
				dst_p[i] = x;
			}
		}
	}
};

class Lut3D : public Lut {
	std::vector<Vector3> m_lut;
	unsigned m_dim;
	float m_scale[3];
	float m_offset[3];
public:
	explicit Lut3D(const Cube &cube) :
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

	void process(const void * const src[3], void * const dst[3], unsigned width) override
	{
		const float *src_r = static_cast<const float *>(src[0]);
		const float *src_g = static_cast<const float *>(src[1]);
		const float *src_b = static_cast<const float *>(src[2]);
		float *dst_r = static_cast<float *>(dst[0]);
		float *dst_g = static_cast<float *>(dst[1]);
		float *dst_b = static_cast<float *>(dst[2]);

		unsigned lut_max = m_dim - 1;

		for (unsigned i = 0; i < width; ++i) {
			float r, g, b;
			float dist_r, dist_g, dist_b;
			unsigned idx_r, idx_g, idx_b;

			Vector3 tri[2][2][2];
			Vector3 interp_result;

			r = src_r[i];
			g = src_g[i];
			b = src_b[i];

			r = (r * m_scale[0] + m_offset[0]) * lut_max;
			g = (g * m_scale[1] + m_offset[1]) * lut_max;
			b = (b * m_scale[2] + m_offset[2]) * lut_max;

			idx_r = std::min(static_cast<unsigned>(r), lut_max);
			idx_g = std::min(static_cast<unsigned>(g), lut_max);
			idx_b = std::min(static_cast<unsigned>(b), lut_max);

			dist_r = r - idx_r;
			dist_g = g - idx_g;
			dist_b = b - idx_b;

			tri[0][0][0] = m_lut[(idx_r + 0) + (idx_g + 0) * m_dim + (idx_b + 0) * m_dim * m_dim];
			tri[0][0][1] = m_lut[(idx_r + 1) + (idx_g + 0) * m_dim + (idx_b + 0) * m_dim * m_dim];
			tri[0][1][0] = m_lut[(idx_r + 0) + (idx_g + 1) * m_dim + (idx_b + 0) * m_dim * m_dim];
			tri[0][1][1] = m_lut[(idx_r + 1) + (idx_g + 1) * m_dim + (idx_b + 0) * m_dim * m_dim];
			tri[1][0][0] = m_lut[(idx_r + 0) + (idx_g + 0) * m_dim + (idx_b + 1) * m_dim * m_dim];
			tri[1][0][1] = m_lut[(idx_r + 1) + (idx_g + 0) * m_dim + (idx_b + 1) * m_dim * m_dim];
			tri[1][1][0] = m_lut[(idx_r + 0) + (idx_g + 1) * m_dim + (idx_b + 1) * m_dim * m_dim];
			tri[1][1][1] = m_lut[(idx_r + 1) + (idx_g + 1) * m_dim + (idx_b + 1) * m_dim * m_dim];

			interp_result = trilinear_interp(tri, dist_b, dist_g, dist_r);
			r = interp_result[0];
			g = interp_result[1];
			b = interp_result[2];

			dst_r[i] = r;
			dst_g[i] = g;
			dst_b[i] = b;
		}
	}
};

} // namespace


std::unique_ptr<Lut> create_lut_impl(const Cube &cube, bool)
{
	std::unique_ptr<Lut> ret;

	if (cube.is_3d) {
		if (!ret) {
			ret.reset(new Lut3D(cube));
		}
	} else {
		if (!ret) {
			ret.reset(new Lut1D(cube));
		}
	}

	return ret;
}

} // namespace timecube
