#ifdef CUBE_X86

#include <cmath>
#include <cstddef>
#include <vector>
#include <smmintrin.h>
#include "cube.h"
#include "lut.h"
#include "lut_x86.h"

#if defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
#elif defined(__GNUC__)
  #define FORCE_INLINE __attribute__((always_inline))
#else
  #define FORCE_INLINE
#endif

namespace timecube {
namespace {

template <class T>
struct AlignedAllocator {
	typedef T value_type;

	AlignedAllocator() = default;

	template <class U>
	AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

	T *allocate(size_t n) const
	{
		void *ptr = _mm_malloc(n * sizeof(T), 16);
		if (!ptr)
			throw std::bad_alloc{};
		return static_cast<T *>(ptr);
	}

	void deallocate(void *ptr, size_t) const noexcept
	{
		_mm_free(ptr);
	}

	bool operator==(const AlignedAllocator &) const noexcept { return true; }
	bool operator!=(const AlignedAllocator &) const noexcept { return false; }
};


// Loads packed vertices for R0-Gx-Bx and R1-Gx-Bx.
static inline FORCE_INLINE void lut3d_load_vertex(const void *lut, const __m128i offset, __m128 &r0, __m128 &g0, __m128 &b0, __m128 &r1, __m128 &g1, __m128 &b1)
{
	__m128 row0, row1, row2, row3;

#define LUT_OFFSET(x) reinterpret_cast<const float *>(static_cast<const unsigned char *>(lut) + (x))
	row0 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 0)));
	row1 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 1)));
	row2 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 2)));
	row3 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 3)));

	_MM_TRANSPOSE4_PS(row0, row1, row2, row3);
	r0 = row0;
	g0 = row1;
	b0 = row2;

	row0 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 0)) + 4);
	row1 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 1)) + 4);
	row2 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 2)) + 4);
	row3 = _mm_load_ps(LUT_OFFSET(_mm_extract_epi32(offset, 3)) + 4);

	_MM_TRANSPOSE4_PS(row0, row1, row2, row3);
	r1 = row0;
	g1 = row1;
	b1 = row2;
#undef LUT_OFFSET
}

static inline FORCE_INLINE __m128 mm_interp_ps(__m128 lo, __m128 hi, __m128 dist)
{
	__m128 x;

	// (1 - x) * a == -x * a + a
	x = _mm_mul_ps(dist, lo);
	x = _mm_sub_ps(lo, x);
	// (-x * a + a) + x * b
	x = _mm_add_ps(_mm_mul_ps(dist, hi), x);

	return x;
}

class Lut3D_SSE41 final : public Lut {
	std::vector<float, AlignedAllocator<float>> m_lut;
	uint_least32_t m_dim;
	float m_scale[3];
	float m_offset[3];
public:
	explicit Lut3D_SSE41(const Cube &cube) :
		m_dim{ cube.n },
		m_scale{},
		m_offset{}
	{
		for (unsigned i = 0; i < 3; ++i) {
			m_scale[i] = (m_dim - 1) / (cube.domain_max[i] - cube.domain_min[i]);
			m_offset[i] = cube.domain_min[i] * m_scale[i];
		}

		// Pad each LUT entry to 16 bytes.
		m_lut.resize(m_dim * m_dim * m_dim * 4);

		for (size_t i = 0; i < m_lut.size() / 4; ++i) {
			m_lut[i * 4 + 0] = cube.lut[i * 3 + 0];
			m_lut[i * 4 + 1] = cube.lut[i * 3 + 1];
			m_lut[i * 4 + 2] = cube.lut[i * 3 + 2];
		}
	}

	void process(const void * const src[3], void * const dst[3], unsigned width) override
	{
		const float *lut = m_lut.data();

		const float *src_r = static_cast<const float *>(src[0]);
		const float *src_g = static_cast<const float *>(src[1]);
		const float *src_b = static_cast<const float *>(src[2]);
		float *dst_r = static_cast<float *>(dst[0]);
		float *dst_g = static_cast<float *>(dst[1]);
		float *dst_b = static_cast<float *>(dst[2]);

		const __m128 scale_r = _mm_set_ps1(m_scale[0]);
		const __m128 scale_g = _mm_set_ps1(m_scale[1]);
		const __m128 scale_b = _mm_set_ps1(m_scale[2]);
		const __m128 offset_r = _mm_set_ps1(m_offset[0]);
		const __m128 offset_g = _mm_set_ps1(m_offset[1]);
		const __m128 offset_b = _mm_set_ps1(m_offset[2]);

		const __m128 lut_max = _mm_set_ps1(std::nextafter(static_cast<float>(m_dim - 1), -INFINITY));
		const __m128i lut_dim = _mm_set1_epi32(m_dim);
		const __m128i lut_dim_sq = _mm_set1_epi32(m_dim * m_dim);

		for (unsigned i = 0; i < width; i += 4) {
			__m128 r = _mm_load_ps(src_r + i);
			__m128 g = _mm_load_ps(src_g + i);
			__m128 b = _mm_load_ps(src_b + i);

			__m128 lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1;
			__m128 tmp0_r, tmp1_r, tmp2_r, tmp3_r;
			__m128 tmp0_g, tmp1_g, tmp2_g, tmp3_g;
			__m128 tmp0_b, tmp1_b, tmp2_b, tmp3_b;

			__m128i idx, idx_r, idx_g, idx_b, idx_base;

			r = _mm_add_ps(_mm_mul_ps(scale_r, r), offset_r);
			g = _mm_add_ps(_mm_mul_ps(scale_g, g), offset_g);
			b = _mm_add_ps(_mm_mul_ps(scale_b, b), offset_b);

			r = _mm_max_ps(r, _mm_setzero_ps());
			r = _mm_min_ps(r, lut_max);

			g = _mm_max_ps(g, _mm_setzero_ps());
			g = _mm_min_ps(g, lut_max);

			b = _mm_max_ps(b, _mm_setzero_ps());
			b = _mm_min_ps(b, lut_max);

			// Base offset.
			idx_r = _mm_cvttps_epi32(r);

			idx_g = _mm_cvttps_epi32(g);
			idx_g = _mm_mullo_epi32(idx_g, lut_dim);

			idx_b = _mm_cvttps_epi32(b);
			idx_b = _mm_mullo_epi32(idx_b, lut_dim_sq);

			idx_base = _mm_add_epi32(idx_r, idx_g);
			idx_base = _mm_add_epi32(idx_base, idx_b);

			// Cube distances.
			r = _mm_sub_ps(r, _mm_floor_ps(r));
			g = _mm_sub_ps(g, _mm_floor_ps(g));
			b = _mm_sub_ps(b, _mm_floor_ps(b));

			// R0-G0-B0 R1-G0-B0
			idx = _mm_slli_epi32(idx_base, 4); // 16 byte entry.

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp0_r = mm_interp_ps(lut_r0, lut_r1, r);
			tmp0_g = mm_interp_ps(lut_g0, lut_g1, r);
			tmp0_b = mm_interp_ps(lut_b0, lut_b1, r);

			// R0-G1-B0 R1-G1-B0
			idx = _mm_add_epi32(idx_base, lut_dim);
			idx = _mm_slli_epi32(idx, 4);

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp1_r = mm_interp_ps(lut_r0, lut_r1, r);
			tmp1_g = mm_interp_ps(lut_g0, lut_g1, r);
			tmp1_b = mm_interp_ps(lut_b0, lut_b1, r);

			// R0-G0-B1 R1-G0-B1
			idx = _mm_add_epi32(idx_base, lut_dim_sq);
			idx = _mm_slli_epi32(idx, 4);

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp2_r = mm_interp_ps(lut_r0, lut_r1, r);
			tmp2_g = mm_interp_ps(lut_g0, lut_g1, r);
			tmp2_b = mm_interp_ps(lut_b0, lut_b1, r);

			// R0-G1-B1 R1-G1-B1
			idx = _mm_add_epi32(idx_base, lut_dim);
			idx = _mm_add_epi32(idx, lut_dim_sq);
			idx = _mm_slli_epi32(idx, 4);

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp3_r = mm_interp_ps(lut_r0, lut_r1, r);
			tmp3_g = mm_interp_ps(lut_g0, lut_g1, r);
			tmp3_b = mm_interp_ps(lut_b0, lut_b1, r);

			// Rx-G0-B0 Rx-G1-B0
			tmp0_r = mm_interp_ps(tmp0_r, tmp1_r, g);
			tmp0_g = mm_interp_ps(tmp0_g, tmp1_g, g);
			tmp0_b = mm_interp_ps(tmp0_b, tmp1_b, g);

			// Rx-G0-B1 Rx-G1-B1
			tmp2_r = mm_interp_ps(tmp2_r, tmp3_r, g);
			tmp2_g = mm_interp_ps(tmp2_g, tmp3_g, g);
			tmp2_b = mm_interp_ps(tmp2_b, tmp3_b, g);

			// Rx-Gx-B0 Rx-Gx-B1
			tmp0_r = mm_interp_ps(tmp0_r, tmp2_r, b);
			tmp0_g = mm_interp_ps(tmp0_g, tmp2_g, b);
			tmp0_b = mm_interp_ps(tmp0_b, tmp2_b, b);

			r = tmp0_r;
			g = tmp0_g;
			b = tmp0_b;

			if (i + 4 > width) {
				alignas(16) float rbuf[4];
				alignas(16) float gbuf[4];
				alignas(16) float bbuf[4];
				_mm_store_ps(rbuf, r);
				_mm_store_ps(gbuf, g);
				_mm_store_ps(bbuf, b);

				for (unsigned ii = i; ii < width; ++ii) {
					dst_r[ii] = rbuf[ii - i];
					dst_g[ii] = gbuf[ii - i];
					dst_b[ii] = bbuf[ii - i];
				}
			} else {
				_mm_store_ps(dst_r + i, r);
				_mm_store_ps(dst_g + i, g);
				_mm_store_ps(dst_b + i, b);
			}
		}
	}
};

} // namespace


std::unique_ptr<Lut> create_lut_impl_sse41(const Cube &cube)
{
	return cube.is_3d ? std::unique_ptr<Lut>(new Lut3D_SSE41{ cube }) : nullptr;
}

} // namespace timecube

#endif // CUBE_X86
