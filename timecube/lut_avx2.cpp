#ifdef CUBE_X86

#include <cmath>
#include <cstddef>
#include <vector>
#include <immintrin.h>
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
		void *ptr = _mm_malloc(n * sizeof(T), 32);
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


static inline FORCE_INLINE __m256 mm256_interp_ps(__m256 lo, __m256 hi, __m256 dist)
{
	__m256 x;

	// (1 - x) * a == -x * a + a
	x = _mm256_fnmadd_ps(dist, lo, lo);
	// (-x * a + a) + x * b
	x = _mm256_fmadd_ps(dist, hi, x);

	return x;
}

// 2x2 transpose of 128-bit elements.
static inline FORCE_INLINE void mm256_transpose2_ps128(__m256 &a, __m256 &b)
{
	__m256 tmp0 = a;
	__m256 tmp1 = b;
	a = _mm256_permute2f128_ps(tmp0, tmp1, 0x20);
	b = _mm256_permute2f128_ps(tmp0, tmp1, 0x31);
}

static inline FORCE_INLINE __m256i lut3d_calculate_index(const __m256 &r, const __m256 &g, const __m256 &b, const __m256i &stride_g, const __m256i &stride_b)
{
	__m256i idx_r, idx_g, idx_b;

	idx_r = _mm256_cvttps_epi32(r);
	idx_r = _mm256_slli_epi32(idx_r, 4); // 16 byte entries.

	idx_g = _mm256_cvttps_epi32(g);
	idx_g = _mm256_mullo_epi32(idx_g, stride_g);

	idx_b = _mm256_cvttps_epi32(b);
	idx_b = _mm256_mullo_epi32(idx_b, stride_b);

	return _mm256_add_epi32(_mm256_add_epi32(idx_r, idx_g), idx_b);
}

// Performs trilinear interpolation on two pixels.
// Returns [R0 G0 B0 xx R1 G1 B1 xx].
static inline FORCE_INLINE __m256 lut3d_trilinear_interp(const void *lut, ptrdiff_t stride_g, ptrdiff_t stride_b, ptrdiff_t idx_lo, ptrdiff_t idx_hi,
                                                         const __m256 &r, const __m256 &g, const __m256 &b)
{
#define LUT_OFFSET(x) reinterpret_cast<const float *>(static_cast<const unsigned char *>(lut) + (x))
	__m256 g0b0_a, g0b1_a, g1b0_a, g1b1_a;
	__m256 g0b0_b, g0b1_b, g1b0_b, g1b1_b;

	g0b0_a = _mm256_loadu_ps(LUT_OFFSET(idx_lo));
	g1b0_a = _mm256_loadu_ps(LUT_OFFSET(idx_lo + stride_g));
	g0b1_a = _mm256_loadu_ps(LUT_OFFSET(idx_lo + stride_b));
	g1b1_a = _mm256_loadu_ps(LUT_OFFSET(idx_lo + stride_b + stride_g));

	g0b0_a = mm256_interp_ps(g0b0_a, g1b0_a, g);
	g0b1_a = mm256_interp_ps(g0b1_a, g1b1_a, g);

	g0b0_a = mm256_interp_ps(g0b0_a, g0b1_a, b);

	g0b0_b = _mm256_loadu_ps(LUT_OFFSET(idx_hi));
	g1b0_b = _mm256_loadu_ps(LUT_OFFSET(idx_hi + stride_g));
	g0b1_b = _mm256_loadu_ps(LUT_OFFSET(idx_hi + stride_b));
	g1b1_b = _mm256_loadu_ps(LUT_OFFSET(idx_hi + stride_b + stride_g));

	g0b0_b = mm256_interp_ps(g0b0_b, g1b0_b, g);
	g0b1_b = mm256_interp_ps(g0b1_b, g1b1_b, g);

	g0b0_b = mm256_interp_ps(g0b0_b, g0b1_b, b);

	mm256_transpose2_ps128(g0b0_a, g0b0_b);
	g0b0_a = mm256_interp_ps(g0b0_a, g0b0_b, r);

	return g0b0_a;
#undef LUT_OFFSET
}

// Converts packed [R0 G0 B0 xx R4 G4 B4 xx] [R1 G1 B1 xx R5 G5 B5 xx] ... to [R1 R2 ...] [G1 G2 ...] [B1 B2 ... ].
static inline FORCE_INLINE void lut3d_unpack_result(const __m256 &result04, const __m256 &result15, const __m256 &result26, const __m256 &result37,
                                                    __m256 &r, __m256 &g, __m256 &b)
{
	__m256 t0 = _mm256_shuffle_ps(result04, result15, 0x44);
	__m256 t1 = _mm256_shuffle_ps(result26, result37, 0x44);
	__m256 t2 = _mm256_shuffle_ps(result04, result15, 0xEE);
	__m256 t3 = _mm256_shuffle_ps(result26, result37, 0xEE);
	
	__m256 tt0 = _mm256_shuffle_ps(t0, t1, 0x88); // r0 r1 r2 r3 | r4 r5 r6 r7
	__m256 tt1 = _mm256_shuffle_ps(t0, t1, 0xDD); // g0 g1 g2 g3 | g4 g5 g6 g7
	__m256 tt2 = _mm256_shuffle_ps(t2, t3, 0x88); // b0 b1 b2 b3 | b4 b5 b6 b7
	// __m256 tt3 = _mm256_shuffle_ps(t2, t3, 0xDD);

	r = tt0;
	g = tt1;
	b = tt2;
}

class Lut3D_AVX2 final : public Lut {
	std::vector<float, AlignedAllocator<float>> m_lut;
	uint_least32_t m_dim;
	float m_scale[3];
	float m_offset[3];
public:
	explicit Lut3D_AVX2(const Cube &cube) :
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

		const __m256 scale_r = _mm256_broadcast_ss(m_scale + 0);
		const __m256 scale_g = _mm256_broadcast_ss(m_scale + 1);
		const __m256 scale_b = _mm256_broadcast_ss(m_scale + 2);
		const __m256 offset_r = _mm256_broadcast_ss(m_offset + 0);
		const __m256 offset_g = _mm256_broadcast_ss(m_offset + 1);
		const __m256 offset_b = _mm256_broadcast_ss(m_offset + 2);

		const __m256 lut_max = _mm256_set1_ps(std::nextafter(static_cast<float>(m_dim - 1), -INFINITY));
		const __m256i lut_stride_g_epi32 = _mm256_set1_epi32(lut_stride_g);
		const __m256i lut_stride_b_epi32 = _mm256_set1_epi32(lut_stride_b);

		const __m256i permute01_mask = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
		const __m256i permute23_mask = _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2);
		const __m256i permute45_mask = _mm256_set_epi32(5, 5, 5, 5, 4, 4, 4, 4);
		const __m256i permute67_mask = _mm256_set_epi32(7, 7, 7, 7, 6, 6, 6, 6);

		for (unsigned i = 0; i < width; i += 8) {
			__m256 r = _mm256_load_ps(src_r + i);
			__m256 g = _mm256_load_ps(src_g + i);
			__m256 b = _mm256_load_ps(src_b + i);

			__m256 result04, result15, result26, result37;
			__m256 rtmp, gtmp, btmp;
			__m256i idx;
			__m128i idx_lo, idx_hi;

			uint32_t idx_scalar_lo, idx_scalar_hi;

			// Input domain remapping.
			r = _mm256_fmadd_ps(r, scale_r, offset_r);
			g = _mm256_fmadd_ps(g, scale_g, offset_g);
			b = _mm256_fmadd_ps(b, scale_b, offset_b);

			r = _mm256_max_ps(r, _mm256_setzero_ps());
			r = _mm256_min_ps(r, lut_max);

			g = _mm256_max_ps(g, _mm256_setzero_ps());
			g = _mm256_min_ps(g, lut_max);

			b = _mm256_max_ps(b, _mm256_setzero_ps());
			b = _mm256_min_ps(b, lut_max);

			// Base offset.
			idx = lut3d_calculate_index(r, g, b, lut_stride_g_epi32, lut_stride_b_epi32);
			idx_lo = _mm256_castsi256_si128(idx);
			idx_hi = _mm256_extracti128_si256(idx, 1);

			// Cube distances.
			r = _mm256_sub_ps(r, _mm256_floor_ps(r));
			g = _mm256_sub_ps(g, _mm256_floor_ps(g));
			b = _mm256_sub_ps(b, _mm256_floor_ps(b));

			// Interpolation.
			rtmp = _mm256_permute_ps(r, _MM_SHUFFLE(0, 0, 0, 0));
			gtmp = _mm256_permute_ps(g, _MM_SHUFFLE(0, 0, 0, 0));
			btmp = _mm256_permute_ps(b, _MM_SHUFFLE(0, 0, 0, 0));
			idx_scalar_lo = _mm_extract_epi32(idx_lo, 0);
			idx_scalar_hi = _mm_extract_epi32(idx_hi, 0);
			result04 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar_lo, idx_scalar_hi, rtmp, gtmp, btmp);

			rtmp = _mm256_permute_ps(r, _MM_SHUFFLE(1, 1, 1, 1));
			gtmp = _mm256_permute_ps(g, _MM_SHUFFLE(1, 1, 1, 1));
			btmp = _mm256_permute_ps(b, _MM_SHUFFLE(1, 1, 1, 1));
			idx_scalar_lo = _mm_extract_epi32(idx_lo, 1);
			idx_scalar_hi = _mm_extract_epi32(idx_hi, 1);
			result15 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar_lo, idx_scalar_hi, rtmp, gtmp, btmp);

			rtmp = _mm256_permute_ps(r, _MM_SHUFFLE(2, 2, 2, 2));
			gtmp = _mm256_permute_ps(g, _MM_SHUFFLE(2, 2, 2, 2));
			btmp = _mm256_permute_ps(b, _MM_SHUFFLE(2, 2, 2, 2));
			idx_scalar_lo = _mm_extract_epi32(idx_lo, 2);
			idx_scalar_hi = _mm_extract_epi32(idx_hi, 2);
			result26 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar_lo, idx_scalar_hi, rtmp, gtmp, btmp);

			rtmp = _mm256_permute_ps(r, _MM_SHUFFLE(3, 3, 3, 3));
			gtmp = _mm256_permute_ps(g, _MM_SHUFFLE(3, 3, 3, 3));
			btmp = _mm256_permute_ps(b, _MM_SHUFFLE(3, 3, 3, 3));
			idx_scalar_lo = _mm_extract_epi32(idx_lo, 3);
			idx_scalar_hi = _mm_extract_epi32(idx_hi, 3);
			result37 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar_lo, idx_scalar_hi, rtmp, gtmp, btmp);

			lut3d_unpack_result(result04, result15, result26, result37, r, g, b);

			if (i + 8 > width) {
				__m256i mask = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
				mask = _mm256_add_epi32(mask, _mm256_set1_epi32(i));
				mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(width), mask);

				_mm256_maskstore_ps(dst_r + i, mask, r);
				_mm256_maskstore_ps(dst_g + i, mask, g);
				_mm256_maskstore_ps(dst_b + i, mask, b);
			} else {
				_mm256_store_ps(dst_r + i, r);
				_mm256_store_ps(dst_g + i, g);
				_mm256_store_ps(dst_b + i, b);
			}
		}
	}
};

} // namespace


std::unique_ptr<Lut> create_lut_impl_avx2(const Cube &cube)
{
	return cube.is_3d ? std::unique_ptr<Lut>(new Lut3D_AVX2{ cube }) : nullptr;
}

} // namespace timecube

#endif // CUBE_X86
