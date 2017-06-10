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


// Unpack LUT data.
//
// In:
// row0: [R0 G0 B0 xx R1 G1 B1 xx]
// row1-7: ...
//
// Out:
// row0: [R0 R0 R0 R0 R0 R0 R0 R0]
// row1: B
// row2: G
// row3: unused
// row4: [R1 R1 R1 R1 R1 R1 R1 R1]
// row5: G
// row6: B
static inline FORCE_INLINE void lut3d_transpose_coeffs(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
	__m256 t0, t1, t2, t3, t4, t5, t6, t7;
	__m256 tt0, tt1, tt2, /* tt3, */ tt4, tt5, tt6 /* , tt7 */;
	t0 = _mm256_unpacklo_ps(row0, row1);
	t1 = _mm256_unpackhi_ps(row0, row1);
	t2 = _mm256_unpacklo_ps(row2, row3);
	t3 = _mm256_unpackhi_ps(row2, row3);
	t4 = _mm256_unpacklo_ps(row4, row5);
	t5 = _mm256_unpackhi_ps(row4, row5);
	t6 = _mm256_unpacklo_ps(row6, row7);
	t7 = _mm256_unpackhi_ps(row6, row7);
	tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
	tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
	tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
	// tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
	tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
	tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
	tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
	// tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
	row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
	row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
	row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
	// row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
	row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
	row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
	row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
	// row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

// Loads packed vertices for R0-Gx-Bx and R1-Gx-Bx.
static inline FORCE_INLINE void lut3d_load_vertex(const void *lut, const __m256i offset, __m256 &r0, __m256 &g0, __m256 &b0, __m256 &r1, __m256 &g1, __m256 &b1)
{
#define LUT_OFFSET(x) reinterpret_cast<const float *>(static_cast<const unsigned char *>(lut) + (x))
	__m128i offset_lo = _mm256_castsi256_si128(offset);
	__m128i offset_hi = _mm256_extracti128_si256(offset, 1);

	__m256 row0 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_lo, 0)));
	__m256 row1 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_lo, 1)));
	__m256 row2 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_lo, 2)));
	__m256 row3 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_lo, 3)));
	__m256 row4 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_hi, 0)));
	__m256 row5 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_hi, 1)));
	__m256 row6 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_hi, 2)));
	__m256 row7 = _mm256_loadu_ps(LUT_OFFSET(_mm_extract_epi32(offset_hi, 3)));

	lut3d_transpose_coeffs(row0, row1, row2, row3, row4, row5, row6, row7);

	r0 = row0;
	g0 = row1;
	b0 = row2;
	r1 = row4;
	g1 = row5;
	b1 = row6;
#undef LUT_OFFSET
}

static inline FORCE_INLINE __m256 mm256_interp_ps(__m256 lo, __m256 hi, __m256 dist)
{
	__m256 x;

	// (1 - x) * a == -x * a + a
	x = _mm256_fnmadd_ps(dist, lo, lo);
	// (-x * a + a) + x * b
	x = _mm256_fmadd_ps(dist, hi, x);

	return x;
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
		const __m256i lut_dim = _mm256_set1_epi32(m_dim);
		const __m256i lut_dim_sq = _mm256_set1_epi32(m_dim * m_dim);

		for (unsigned i = 0; i < width; i += 8) {
			__m256 r = _mm256_load_ps(src_r + i);
			__m256 g = _mm256_load_ps(src_g + i);
			__m256 b = _mm256_load_ps(src_b + i);

			__m256 lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1;
			__m256 tmp0_r, tmp1_r, tmp2_r, tmp3_r;
			__m256 tmp0_g, tmp1_g, tmp2_g, tmp3_g;
			__m256 tmp0_b, tmp1_b, tmp2_b, tmp3_b;

			__m256i idx, idx_r, idx_g, idx_b, idx_base;

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
			idx_r = _mm256_cvttps_epi32(r);

			idx_g = _mm256_cvttps_epi32(g);
			idx_g = _mm256_mullo_epi32(idx_g, lut_dim);

			idx_b = _mm256_cvttps_epi32(b);
			idx_b = _mm256_mullo_epi32(idx_b, lut_dim_sq);

			idx_base = _mm256_add_epi32(idx_r, idx_g);
			idx_base = _mm256_add_epi32(idx_base, idx_b);

			// Cube distances.
			r = _mm256_sub_ps(r, _mm256_floor_ps(r));
			g = _mm256_sub_ps(g, _mm256_floor_ps(g));
			b = _mm256_sub_ps(b, _mm256_floor_ps(b));

			// R0-G0-B0 R1-G0-B0
			idx = _mm256_slli_epi32(idx_base, 4); // 16 byte entry.

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp0_r = mm256_interp_ps(lut_r0, lut_r1, r);
			tmp0_g = mm256_interp_ps(lut_g0, lut_g1, r);
			tmp0_b = mm256_interp_ps(lut_b0, lut_b1, r);

			// R0-G1-B0 R1-G1-B0
			idx = _mm256_add_epi32(idx_base, lut_dim);
			idx = _mm256_slli_epi32(idx, 4);

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp1_r = mm256_interp_ps(lut_r0, lut_r1, r);
			tmp1_g = mm256_interp_ps(lut_g0, lut_g1, r);
			tmp1_b = mm256_interp_ps(lut_b0, lut_b1, r);

			// R0-G0-B1 R1-G0-B1
			idx = _mm256_add_epi32(idx_base, lut_dim_sq);
			idx = _mm256_slli_epi32(idx, 4);

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp2_r = mm256_interp_ps(lut_r0, lut_r1, r);
			tmp2_g = mm256_interp_ps(lut_g0, lut_g1, r);
			tmp2_b = mm256_interp_ps(lut_b0, lut_b1, r);

			// R0-G1-B1 R1-G1-B1
			idx = _mm256_add_epi32(idx_base, lut_dim);
			idx = _mm256_add_epi32(idx, lut_dim_sq);
			idx = _mm256_slli_epi32(idx, 4);

			lut3d_load_vertex(lut, idx, lut_r0, lut_g0, lut_b0, lut_r1, lut_g1, lut_b1);
			tmp3_r = mm256_interp_ps(lut_r0, lut_r1, r);
			tmp3_g = mm256_interp_ps(lut_g0, lut_g1, r);
			tmp3_b = mm256_interp_ps(lut_b0, lut_b1, r);

			// Rx-G0-B0 Rx-G1-B0
			tmp0_r = mm256_interp_ps(tmp0_r, tmp1_r, g);
			tmp0_g = mm256_interp_ps(tmp0_g, tmp1_g, g);
			tmp0_b = mm256_interp_ps(tmp0_b, tmp1_b, g);

			// Rx-G0-B1 Rx-G1-B1
			tmp2_r = mm256_interp_ps(tmp2_r, tmp3_r, g);
			tmp2_g = mm256_interp_ps(tmp2_g, tmp3_g, g);
			tmp2_b = mm256_interp_ps(tmp2_b, tmp3_b, g);

			// Rx-Gx-B0 Rx-Gx-B1
			tmp0_r = mm256_interp_ps(tmp0_r, tmp2_r, b);
			tmp0_g = mm256_interp_ps(tmp0_g, tmp2_g, b);
			tmp0_b = mm256_interp_ps(tmp0_b, tmp2_b, b);

			r = tmp0_r;
			g = tmp0_g;
			b = tmp0_b;

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
