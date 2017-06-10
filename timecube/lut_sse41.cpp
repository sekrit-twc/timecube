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

static inline FORCE_INLINE __m128i lut3d_calculate_index(const __m128 &r, const __m128 &g, const __m128 &b, const __m128i &stride_g, const __m128i &stride_b)
{
	__m128i idx_r, idx_g, idx_b;

	idx_r = _mm_cvttps_epi32(r);
	idx_r = _mm_slli_epi32(idx_r, 4); // 16 byte entries.

	idx_g = _mm_cvttps_epi32(g);
	idx_g = _mm_mullo_epi32(idx_g, stride_g);

	idx_b = _mm_cvttps_epi32(b);
	idx_b = _mm_mullo_epi32(idx_b, stride_b);

	return _mm_add_epi32(_mm_add_epi32(idx_r, idx_g), idx_b);
}

// Performs trilinear interpolation on one pixels.
// Returns [R G B x].
static inline FORCE_INLINE __m128 lut3d_trilinear_interp(const void *lut, ptrdiff_t stride_g, ptrdiff_t stride_b, ptrdiff_t idx,
                                                         const __m128 &r, const __m128 &g, const __m128 &b)
{
#define LUT_OFFSET(x) reinterpret_cast<const float *>(static_cast<const unsigned char *>(lut) + (x))
	__m128 r0g0b0, r1g0b0, r0g1b0, r1g1b0, r0g0b1, r1g0b1, r0g1b1, r1g1b1;

	r0g0b0 = _mm_load_ps(LUT_OFFSET(idx));
	r1g0b0 = _mm_load_ps(LUT_OFFSET(idx + 16));
	r0g1b0 = _mm_load_ps(LUT_OFFSET(idx + stride_g));
	r1g1b0 = _mm_load_ps(LUT_OFFSET(idx + stride_g + 16));
	r0g0b1 = _mm_load_ps(LUT_OFFSET(idx + stride_b));
	r1g0b1 = _mm_load_ps(LUT_OFFSET(idx + stride_b + 16));
	r0g1b1 = _mm_load_ps(LUT_OFFSET(idx + stride_g + stride_b));
	r1g1b1 = _mm_load_ps(LUT_OFFSET(idx + stride_g + stride_b + 16));

	r0g0b0 = mm_interp_ps(r0g0b0, r1g0b0, r);
	r0g1b0 = mm_interp_ps(r0g1b0, r1g1b0, r);
	r0g0b1 = mm_interp_ps(r0g0b1, r1g0b1, r);
	r0g1b1 = mm_interp_ps(r0g1b1, r1g1b1, r);

	r0g0b0 = mm_interp_ps(r0g0b0, r0g1b0, g);
	r0g0b1 = mm_interp_ps(r0g0b1, r0g1b1, g);

	r0g0b0 = mm_interp_ps(r0g0b0, r0g0b1, b);
	return r0g0b0;
#undef LUT_OFFSET
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
		uint32_t lut_stride_g = m_dim * sizeof(float) * 4;
		uint32_t lut_stride_b = m_dim * m_dim * sizeof(float) * 4;

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
		const __m128i lut_stride_g_epi32 = _mm_set1_epi32(lut_stride_g);
		const __m128i lut_stride_b_epi32 = _mm_set1_epi32(lut_stride_b);

		for (unsigned i = 0; i < width; i += 4) {
			__m128 r = _mm_load_ps(src_r + i);
			__m128 g = _mm_load_ps(src_g + i);
			__m128 b = _mm_load_ps(src_b + i);

			__m128 result0, result1, result2, result3;
			__m128 rtmp, gtmp, btmp;
			__m128i idx;

			uint32_t idx_scalar;

			// Input domain remapping.
			r = _mm_add_ps(_mm_mul_ps(scale_r, r), offset_r);
			g = _mm_add_ps(_mm_mul_ps(scale_g, g), offset_g);
			b = _mm_add_ps(_mm_mul_ps(scale_b, b), offset_b);

			r = _mm_max_ps(r, _mm_setzero_ps());
			r = _mm_min_ps(r, lut_max);

			g = _mm_max_ps(g, _mm_setzero_ps());
			g = _mm_min_ps(g, lut_max);

			b = _mm_max_ps(b, _mm_setzero_ps());
			b = _mm_min_ps(b, lut_max);

			idx = lut3d_calculate_index(r, g, b, lut_stride_g_epi32, lut_stride_b_epi32);

			// Cube distances.
			r = _mm_sub_ps(r, _mm_floor_ps(r));
			g = _mm_sub_ps(g, _mm_floor_ps(g));
			b = _mm_sub_ps(b, _mm_floor_ps(b));

			// Interpolation.
			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 0));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(0, 0, 0, 0));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 0, 0));
			idx_scalar = _mm_extract_epi32(idx, 0);
			result0 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 1, 1, 1));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(1, 1, 1, 1));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1));
			idx_scalar = _mm_extract_epi32(idx, 1);
			result1 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 2, 2, 2));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(2, 2, 2, 2));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 2, 2));
			idx_scalar = _mm_extract_epi32(idx, 2);
			result2 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 3, 3, 3));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(3, 3, 3, 3));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3));
			idx_scalar = _mm_extract_epi32(idx, 3);
			result3 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar, rtmp, gtmp, btmp);

			_MM_TRANSPOSE4_PS(result0, result1, result2, result3);
			r = result0;
			g = result1;
			b = result2;

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
