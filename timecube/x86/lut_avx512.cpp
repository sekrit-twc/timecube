#ifdef CUBE_X86

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
		void *ptr = _mm_malloc(n * sizeof(T), 64);
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


static inline FORCE_INLINE __m512 mm512_interp_ps(__m512 lo, __m512 hi, __m512 dist)
{
	__m512 x;

	// (1 - x) * a == -x * a + a
	x = _mm512_fnmadd_ps(dist, lo, lo);
	// (-x * a + a) + x * b
	x = _mm512_fmadd_ps(dist, hi, x);

	return x;
}

// Transpose 2x2 128-bit elements within upper and lower 256-bit lanes.
static inline FORCE_INLINE void mm512_transpose2_ps128(__m512 &a, __m512 &b)
{
	__m512 tmp0 = a;
	__m512 tmp1 = b;
	a = _mm512_shuffle_f32x4(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0));
	b = _mm512_shuffle_f32x4(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1));
}

static inline FORCE_INLINE __m512i lut3d_calculate_index(const __m512 &r, const __m512 &g, const __m512 &b, ptrdiff_t stride_g, ptrdiff_t stride_b)
{
	__m512i idx_r, idx_g, idx_b;

	idx_r = _mm512_cvttps_epi32(r);
	idx_r = _mm512_slli_epi32(idx_r, 4); // 16 byte entries.

	idx_g = _mm512_cvttps_epi32(g);
	idx_g = _mm512_mullo_epi32(idx_g, _mm512_set1_epi32(static_cast<uint32_t>(stride_g)));

	idx_b = _mm512_cvttps_epi32(b);
	idx_b = _mm512_mullo_epi32(idx_b, _mm512_set1_epi32(static_cast<uint32_t>(stride_b)));

	return _mm512_add_epi32(_mm512_add_epi32(idx_r, idx_g), idx_b);
}


// Performs trilinear interpolation on four pixels.
// Returns [R0 G0 B0 xx R1 G1 B1 xx R2 G2 B2 x R3 G3 B3 x].
static inline FORCE_INLINE __m512 lut3d_trilinear_interp(const void *lut, ptrdiff_t stride_g, ptrdiff_t stride_b,
                                                         ptrdiff_t idx_lolo, ptrdiff_t idx_lohi, ptrdiff_t idx_hilo, ptrdiff_t idx_hihi,
                                                         __m512 r, __m512 g, __m512 b)
{
#define LUT_OFFSET(x) reinterpret_cast<const float *>(static_cast<const unsigned char *>(lut) + (x))
	__m512 g_lo = _mm512_shuffle_f32x4(g, g, _MM_SHUFFLE(1, 1, 0, 0));
	__m512 b_lo = _mm512_shuffle_f32x4(b, b, _MM_SHUFFLE(1, 1, 0, 0));
	__m512 g_hi = _mm512_shuffle_f32x4(g, g, _MM_SHUFFLE(3, 3, 2, 2));
	__m512 b_hi = _mm512_shuffle_f32x4(b, b, _MM_SHUFFLE(3, 3, 2, 2));

	__m512 g0b0_a, g0b1_a, g1b0_a, g1b1_a;
	__m512 g0b0_b, g0b1_b, g1b0_b, g1b1_b;

	g0b0_a = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_lolo))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_lohi)), 1);
	g1b0_a = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_lolo + stride_g))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_lohi + stride_g)), 1);
	g0b1_a = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_lolo + stride_b))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_lohi + stride_b)), 1);
	g1b1_a = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_lolo + stride_b + stride_g))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_lohi + stride_b + stride_g)), 1);

	g0b0_a = mm512_interp_ps(g0b0_a, g1b0_a, g_lo);
	g0b1_a = mm512_interp_ps(g0b1_a, g1b1_a, g_lo);

	g0b0_a = mm512_interp_ps(g0b0_a, g0b1_a, b_lo);

	g0b0_b = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_hilo))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_hihi)), 1);
	g1b0_b = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_hilo + stride_g))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_hihi + stride_g)), 1);
	g0b1_b = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_hilo + stride_b))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_hihi + stride_b)), 1);
	g1b1_b = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(LUT_OFFSET(idx_hilo + stride_b + stride_g))),
	                            _mm256_loadu_ps(LUT_OFFSET(idx_hihi + stride_b + stride_g)), 1);

	g0b0_b = mm512_interp_ps(g0b0_b, g1b0_b, g_hi);
	g0b1_b = mm512_interp_ps(g0b1_b, g1b1_b, g_hi);

	g0b0_b = mm512_interp_ps(g0b0_b, g0b1_b, b_hi);

	mm512_transpose2_ps128(g0b0_a, g0b0_b);
	g0b0_a = mm512_interp_ps(g0b0_a, g0b0_b, r);

	return g0b0_a;
#undef LUT_OFFSET
}

static inline FORCE_INLINE void minmax(__m512 &x, __m512 &y)
{
	__m512 minval = _mm512_min_ps(x, y);
	__m512 maxval = _mm512_max_ps(x, y);
	x = minval;
	y = maxval;
}

// Identify sub-tetra containing each pixel.
// Writes-back normalized (sorted) pixel coordinates into r, g, b.
// Returns absolute displacements from vertexes (0,0,0) and (1,1,1) in disp1, disp2.
static inline FORCE_INLINE void lut3d_tetra_classify(__m512 &r, __m512 &g, __m512 &b, __m512i &disp1, __m512i &disp2,
                                                     ptrdiff_t stride_g, ptrdiff_t stride_b)
{
	__m512 x = r, y = g, z = b;
	__m512i stride_r_epi32 = _mm512_set1_epi32(sizeof(float) * 4);
	__m512i stride_g_epi32 = _mm512_set1_epi32(static_cast<uint32_t>(stride_g));
	__m512i stride_b_epi32 = _mm512_set1_epi32(static_cast<uint32_t>(stride_b));
	__m512i tmp;

	// Sort.
	minmax(x, z);
	minmax(x, y);
	minmax(y, z);

	tmp = stride_b_epi32;
	tmp = _mm512_mask_mov_epi32(tmp, _mm512_cmpeq_epu32_mask(_mm512_castps_si512(z), _mm512_castps_si512(g)), stride_g_epi32);
	tmp = _mm512_mask_mov_epi32(tmp, _mm512_cmpeq_epu32_mask(_mm512_castps_si512(z), _mm512_castps_si512(r)), stride_r_epi32);
	disp1 = tmp;

	tmp = stride_b_epi32;
	tmp = _mm512_mask_mov_epi32(tmp, _mm512_cmpeq_epu32_mask(_mm512_castps_si512(x), _mm512_castps_si512(g)), stride_g_epi32);
	tmp = _mm512_mask_mov_epi32(tmp, _mm512_cmpeq_epu32_mask(_mm512_castps_si512(x), _mm512_castps_si512(r)), stride_r_epi32);
	disp2 = tmp;

	r = x;
	g = y;
	b = z;
}

// Performs tetrahedral interpolation on four pixels.
// Returns [R0 G0 B0 xx R1 G1 B1 xx R2 G2 B2 x R3 G3 B3 x].
static inline FORCE_INLINE __m512 lut3d_tetra_interp(const void *lut,
                                                     ptrdiff_t idx_base_lolo, ptrdiff_t idx_base_lohi, ptrdiff_t idx_base_hilo, ptrdiff_t idx_base_hihi,
                                                     ptrdiff_t idx_diag_lolo, ptrdiff_t idx_diag_lohi, ptrdiff_t idx_diag_hilo, ptrdiff_t idx_diag_hihi,
                                                     ptrdiff_t idx_disp1_lolo, ptrdiff_t idx_disp1_lohi, ptrdiff_t idx_disp1_hilo, ptrdiff_t idx_disp1_hihi,
                                                     ptrdiff_t idx_disp2_lolo, ptrdiff_t idx_disp2_lohi, ptrdiff_t idx_disp2_hilo, ptrdiff_t idx_disp2_hihi,
                                                     __m512 x, __m512 y, __m512 z)
{
#define LUT_OFFSET(x) reinterpret_cast<const float *>(static_cast<const unsigned char *>(lut) + (x))
	__m512 v0 = _mm512_castps128_ps512(_mm_load_ps(LUT_OFFSET(idx_base_lolo)));
	v0 = _mm512_insertf32x4(v0, _mm_load_ps(LUT_OFFSET(idx_base_lohi)), 1);
	v0 = _mm512_insertf32x4(v0, _mm_load_ps(LUT_OFFSET(idx_base_hilo)), 2);
	v0 = _mm512_insertf32x4(v0, _mm_load_ps(LUT_OFFSET(idx_base_hihi)), 3);

	__m512 v1 = _mm512_castps128_ps512(_mm_load_ps(LUT_OFFSET(idx_diag_lolo)));
	v1 = _mm512_insertf32x4(v1, _mm_load_ps(LUT_OFFSET(idx_diag_lohi)), 1);
	v1 = _mm512_insertf32x4(v1, _mm_load_ps(LUT_OFFSET(idx_diag_hilo)), 2);
	v1 = _mm512_insertf32x4(v1, _mm_load_ps(LUT_OFFSET(idx_diag_hihi)), 3);

	__m512 v2 = _mm512_castps128_ps512(_mm_load_ps(LUT_OFFSET(idx_disp1_lolo)));
	v2 = _mm512_insertf32x4(v2, _mm_load_ps(LUT_OFFSET(idx_disp1_lohi)), 1);
	v2 = _mm512_insertf32x4(v2, _mm_load_ps(LUT_OFFSET(idx_disp1_hilo)), 2);
	v2 = _mm512_insertf32x4(v2, _mm_load_ps(LUT_OFFSET(idx_disp1_hihi)), 3);

	__m512 v3 = _mm512_castps128_ps512(_mm_load_ps(LUT_OFFSET(idx_disp2_lolo)));
	v3 = _mm512_insertf32x4(v3, _mm_load_ps(LUT_OFFSET(idx_disp2_lohi)), 1);
	v3 = _mm512_insertf32x4(v3, _mm_load_ps(LUT_OFFSET(idx_disp2_hilo)), 2);
	v3 = _mm512_insertf32x4(v3, _mm_load_ps(LUT_OFFSET(idx_disp2_hihi)), 3);

	// (1 - z) * v0 + x * v1 + (z - y) * v2 + (y - x) * v3
	//   ==>
	// v0 - z * v0 + x * v1 + z * v2 - y * v2 + y * v3 - x * v3
	__m512 result = v0;
	result = _mm512_fnmadd_ps(z, v0, result);
	result = _mm512_fmadd_ps(x, v1, result);
	result = _mm512_fmadd_ps(z, v2, result);
	result = _mm512_fnmadd_ps(y, v2, result);
	result = _mm512_fmadd_ps(y, v3, result);
	result = _mm512_fnmadd_ps(x, v3, result);
	return result;
#undef LUT_OFFSET
}

// Converts packed [R0 G0 B0 xx R4 G4 B4 xx R8 G8 B8 xx Rc Gc Bc xx] ... to [R1 R2 ...] [G1 G2 ...] [B1 B2 ...].
static inline FORCE_INLINE void lut3d_unpack_result(const __m512 &result048c, const __m512 &result159d, const __m512 &result26ae, const __m512 &result37bf,
                                                    __m512 &r, __m512 &g, __m512 &b)
{
	__m512 t0 = _mm512_shuffle_ps(result048c, result159d, 0x44);
	__m512 t1 = _mm512_shuffle_ps(result26ae, result37bf, 0x44);
	__m512 t2 = _mm512_shuffle_ps(result048c, result159d, 0xEE);
	__m512 t3 = _mm512_shuffle_ps(result26ae, result37bf, 0xEE);

	__m512 tt0 = _mm512_shuffle_ps(t0, t1, 0x88); // r0 r1 r2 r3 | r4 r5 r6 r7
	__m512 tt1 = _mm512_shuffle_ps(t0, t1, 0xDD); // g0 g1 g2 g3 | g4 g5 g6 g7
	__m512 tt2 = _mm512_shuffle_ps(t2, t3, 0x88); // b0 b1 b2 b3 | b4 b5 b6 b7
	// __m512 tt3 = _mm512_shuffle_ps(t2, t3, 0xDD);

	r = tt0;
	g = tt1;
	b = tt2;
}


class Lut3DFilter_AVX512 : public Lut3DFilter {
protected:
	std::vector<float, AlignedAllocator<float>> m_lut;
	uint32_t m_dim;
	float m_scale[3];
	float m_offset[3];

	Lut3DFilter_AVX512(const Cube &cube, unsigned width, unsigned height) :
		Lut3DFilter(width, height),
		m_dim{ cube.n },
		m_scale{},
		m_offset{}
	{
		for (unsigned i = 0; i < 3; ++i) {
			m_scale[i] = (m_dim - 1) / (cube.domain_max[i] - cube.domain_min[i]);
			m_offset[i] = -cube.domain_min[i] * m_scale[i];
		}

		// Pad each LUT entry to 16 bytes.
		m_lut.resize(m_dim * m_dim * m_dim * 4);

		for (size_t i = 0; i < m_lut.size() / 4; ++i) {
			m_lut[i * 4 + 0] = cube.lut[i * 3 + 0];
			m_lut[i * 4 + 1] = cube.lut[i * 3 + 1];
			m_lut[i * 4 + 2] = cube.lut[i * 3 + 2];
		}

		m_desc.alignment_mask = 0x15;
	}
};

class TrilinearFilter_AVX512 final : public Lut3DFilter_AVX512 {
public:
	TrilinearFilter_AVX512(const Cube &cube, unsigned width, unsigned height) : Lut3DFilter_AVX512(cube, width, height) {}

	void process(const graphengine::BufferDescriptor in[], const graphengine::BufferDescriptor out[],
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *lut = m_lut.data();
		uint32_t lut_stride_g = m_dim * sizeof(float) * 4;
		uint32_t lut_stride_b = m_dim * m_dim * sizeof(float) * 4;

		const float *src_r = in[0].get_line<float>(i);
		const float *src_g = in[1].get_line<float>(i);
		const float *src_b = in[2].get_line<float>(i);
		float *dst_r = out[0].get_line<float>(i);
		float *dst_g = out[1].get_line<float>(i);
		float *dst_b = out[2].get_line<float>(i);

		const __m512 scale_r = _mm512_set1_ps(m_scale[0]);
		const __m512 scale_g = _mm512_set1_ps(m_scale[1]);
		const __m512 scale_b = _mm512_set1_ps(m_scale[2]);
		const __m512 offset_r = _mm512_set1_ps(m_offset[0]);
		const __m512 offset_g = _mm512_set1_ps(m_offset[1]);
		const __m512 offset_b = _mm512_set1_ps(m_offset[2]);

		const __m512 lut_max = _mm512_set1_ps(std::nextafter(static_cast<float>(m_dim - 1), -INFINITY));

		for (unsigned i = left; i < right; i += 16) {
			__m512 r = _mm512_load_ps(src_r + i);
			__m512 g = _mm512_load_ps(src_g + i);
			__m512 b = _mm512_load_ps(src_b + i);

			__m512 result048c, result159d, result26ae, result37bf;
			__m512 rtmp, gtmp, btmp;
			__m512i idx;
			__m128i idx_lolo, idx_lohi, idx_hilo, idx_hihi;

			size_t idx_scalar_lolo, idx_scalar_lohi, idx_scalar_hilo, idx_scalar_hihi;

			// Input domain remapping.
			r = _mm512_fmadd_ps(r, scale_r, offset_r);
			g = _mm512_fmadd_ps(g, scale_g, offset_g);
			b = _mm512_fmadd_ps(b, scale_b, offset_b);

			r = _mm512_max_ps(r, _mm512_setzero_ps());
			r = _mm512_min_ps(r, lut_max);

			g = _mm512_max_ps(g, _mm512_setzero_ps());
			g = _mm512_min_ps(g, lut_max);

			b = _mm512_max_ps(b, _mm512_setzero_ps());
			b = _mm512_min_ps(b, lut_max);

			// Base offset.
			idx = lut3d_calculate_index(r, g, b, lut_stride_g, lut_stride_b);
			idx_lolo = _mm512_castsi512_si128(idx);
			idx_lohi = _mm512_extracti32x4_epi32(idx, 1);
			idx_hilo = _mm512_extracti32x4_epi32(idx, 2);
			idx_hihi = _mm512_extracti32x4_epi32(idx, 3);

			// Cube distances.
			r = _mm512_sub_ps(r, _mm512_roundscale_ps(r, 1));
			g = _mm512_sub_ps(g, _mm512_roundscale_ps(g, 1));
			b = _mm512_sub_ps(b, _mm512_roundscale_ps(b, 1));

			// Interpolation.
#if SIZE_MAX >= UINT64_MAX
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi64((x), (idx) / 2)
  #define EXTRACT_ODD(out, x, idx) ((out) >> 32)
#else
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi32((x), (idx))
  #define EXTRACT_ODD(out, x, idx) _mm_extract_epi32((x), (idx))
#endif

#define INDICES idx_scalar_lolo & 0xFFFFFFFFU, idx_scalar_lohi & 0xFFFFFFFFU, idx_scalar_hilo & 0xFFFFFFFFU, idx_scalar_hihi & 0xFFFFFFFFU
			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(0, 0, 0, 0));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(0, 0, 0, 0));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(0, 0, 0, 0));
			idx_scalar_lolo = EXTRACT_EVEN(idx_scalar_lolo, idx_lolo, 0);
			idx_scalar_lohi = EXTRACT_EVEN(idx_scalar_lohi, idx_lohi, 0);
			idx_scalar_hilo = EXTRACT_EVEN(idx_scalar_hilo, idx_hilo, 0);
			idx_scalar_hihi = EXTRACT_EVEN(idx_scalar_hihi, idx_hihi, 0);
			result048c = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(1, 1, 1, 1));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(1, 1, 1, 1));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(1, 1, 1, 1));
			idx_scalar_lolo = EXTRACT_ODD(idx_scalar_lolo, idx_lolo, 1);
			idx_scalar_lohi = EXTRACT_ODD(idx_scalar_lohi, idx_lohi, 1);
			idx_scalar_hilo = EXTRACT_ODD(idx_scalar_hilo, idx_hilo, 1);
			idx_scalar_hihi = EXTRACT_ODD(idx_scalar_hihi, idx_hihi, 1);
			result159d = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(2, 2, 2, 2));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(2, 2, 2, 2));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(2, 2, 2, 2));
			idx_scalar_lolo = EXTRACT_EVEN(idx_scalar_lolo, idx_lolo, 2);
			idx_scalar_lohi = EXTRACT_EVEN(idx_scalar_lohi, idx_lohi, 2);
			idx_scalar_hilo = EXTRACT_EVEN(idx_scalar_hilo, idx_hilo, 2);
			idx_scalar_hihi = EXTRACT_EVEN(idx_scalar_hihi, idx_hihi, 2);
			result26ae = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(3, 3, 3, 3));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(3, 3, 3, 3));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(3, 3, 3, 3));
			idx_scalar_lolo = EXTRACT_ODD(idx_scalar_lolo, idx_lolo, 3);
			idx_scalar_lohi = EXTRACT_ODD(idx_scalar_lohi, idx_lohi, 3);
			idx_scalar_hilo = EXTRACT_ODD(idx_scalar_hilo, idx_hilo, 3);
			idx_scalar_hihi = EXTRACT_ODD(idx_scalar_hihi, idx_hihi, 3);
			result37bf = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, INDICES, rtmp, gtmp, btmp);
#undef INDICES
#undef EXTRACT_ODD
#undef EXTRACT_EVEN
			lut3d_unpack_result(result048c, result159d, result26ae, result37bf, r, g, b);

			_mm512_store_ps(dst_r + i, r);
			_mm512_store_ps(dst_g + i, g);
			_mm512_store_ps(dst_b + i, b);
		}
	}
};

class TetrahedralFilter_AVX512 final : public Lut3DFilter_AVX512 {
public:
	TetrahedralFilter_AVX512(const Cube &cube, unsigned width, unsigned height) : Lut3DFilter_AVX512(cube, width, height) {}

	void process(const graphengine::BufferDescriptor in[], const graphengine::BufferDescriptor out[],
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *lut = m_lut.data();
		uint32_t lut_stride_g = m_dim * sizeof(float) * 4;
		uint32_t lut_stride_b = m_dim * m_dim * sizeof(float) * 4;
		uint32_t lut_stride_diag = lut_stride_g + lut_stride_b + sizeof(float) * 4;

		const float *src_r = in[0].get_line<float>(i);
		const float *src_g = in[1].get_line<float>(i);
		const float *src_b = in[2].get_line<float>(i);
		float *dst_r = out[0].get_line<float>(i);
		float *dst_g = out[1].get_line<float>(i);
		float *dst_b = out[2].get_line<float>(i);

		const __m512 scale_r = _mm512_set1_ps(m_scale[0]);
		const __m512 scale_g = _mm512_set1_ps(m_scale[1]);
		const __m512 scale_b = _mm512_set1_ps(m_scale[2]);
		const __m512 offset_r = _mm512_set1_ps(m_offset[0]);
		const __m512 offset_g = _mm512_set1_ps(m_offset[1]);
		const __m512 offset_b = _mm512_set1_ps(m_offset[2]);

		const __m512 lut_max = _mm512_set1_ps(std::nextafter(static_cast<float>(m_dim - 1), -INFINITY));

		for (unsigned i = left; i < right; i += 16) {
			__m512 r = _mm512_load_ps(src_r + i);
			__m512 g = _mm512_load_ps(src_g + i);
			__m512 b = _mm512_load_ps(src_b + i);

			__m512 result048c, result159d, result26ae, result37bf;
			__m512 rtmp, gtmp, btmp;
			__m512i idx, diag, disp1, disp2;
			__m128i idx_lolo, idx_lohi, idx_hilo, idx_hihi;
			__m128i diag_lolo, diag_lohi, diag_hilo, diag_hihi;
			__m128i disp1_lolo, disp1_lohi, disp1_hilo, disp1_hihi;
			__m128i disp2_lolo, disp2_lohi, disp2_hilo, disp2_hihi;

			size_t idx_scalar_lolo, idx_scalar_lohi, idx_scalar_hilo, idx_scalar_hihi;
			size_t diag_scalar_lolo, diag_scalar_lohi, diag_scalar_hilo, diag_scalar_hihi;
			size_t disp1_scalar_lolo, disp1_scalar_lohi, disp1_scalar_hilo, disp1_scalar_hihi;
			size_t disp2_scalar_lolo, disp2_scalar_lohi, disp2_scalar_hilo, disp2_scalar_hihi;

			// Input domain remapping.
			r = _mm512_fmadd_ps(r, scale_r, offset_r);
			g = _mm512_fmadd_ps(g, scale_g, offset_g);
			b = _mm512_fmadd_ps(b, scale_b, offset_b);

			r = _mm512_max_ps(r, _mm512_setzero_ps());
			r = _mm512_min_ps(r, lut_max);

			g = _mm512_max_ps(g, _mm512_setzero_ps());
			g = _mm512_min_ps(g, lut_max);

			b = _mm512_max_ps(b, _mm512_setzero_ps());
			b = _mm512_min_ps(b, lut_max);

			// Base offset.
			idx = lut3d_calculate_index(r, g, b, lut_stride_g, lut_stride_b);
			diag = _mm512_add_epi32(idx, _mm512_set1_epi32(lut_stride_diag));

			idx_lolo = _mm512_castsi512_si128(idx);
			idx_lohi = _mm512_extracti32x4_epi32(idx, 1);
			idx_hilo = _mm512_extracti32x4_epi32(idx, 2);
			idx_hihi = _mm512_extracti32x4_epi32(idx, 3);

			diag_lolo = _mm512_castsi512_si128(diag);
			diag_lohi = _mm512_extracti32x4_epi32(diag, 1);
			diag_hilo = _mm512_extracti32x4_epi32(diag, 2);
			diag_hihi = _mm512_extracti32x4_epi32(diag, 3);

			// Cube distances.
			r = _mm512_sub_ps(r, _mm512_roundscale_ps(r, 1));
			g = _mm512_sub_ps(g, _mm512_roundscale_ps(g, 1));
			b = _mm512_sub_ps(b, _mm512_roundscale_ps(b, 1));

			// Classification.
			lut3d_tetra_classify(r, g, b, disp1, disp2, lut_stride_g, lut_stride_b);
			disp1 = _mm512_add_epi32(idx, disp1);
			disp2 = _mm512_sub_epi32(diag, disp2);

			disp1_lolo = _mm512_castsi512_si128(disp1);
			disp1_lohi = _mm512_extracti32x4_epi32(disp1, 1);
			disp1_hilo = _mm512_extracti32x4_epi32(disp1, 2);
			disp1_hihi = _mm512_extracti32x4_epi32(disp1, 3);

			disp2_lolo = _mm512_castsi512_si128(disp2);
			disp2_lohi = _mm512_extracti32x4_epi32(disp2, 1);
			disp2_hilo = _mm512_extracti32x4_epi32(disp2, 2);
			disp2_hihi = _mm512_extracti32x4_epi32(disp2, 3);

			// Interpolation.
#if SIZE_MAX >= UINT64_MAX
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi64((x), (idx) / 2)
  #define EXTRACT_ODD(out, x, idx) ((out) >> 32)
#else
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi32((x), (idx))
  #define EXTRACT_ODD(out, x, idx) _mm_extract_epi32((x), (idx))
#endif

#define INDICES idx_scalar_lolo & 0xFFFFFFFFU, idx_scalar_lohi & 0xFFFFFFFFU, idx_scalar_hilo & 0xFFFFFFFFU, idx_scalar_hihi & 0xFFFFFFFFU, \
  diag_scalar_lolo & 0xFFFFFFFFU, diag_scalar_lohi & 0xFFFFFFFFU, diag_scalar_hilo & 0xFFFFFFFFU, diag_scalar_hihi & 0xFFFFFFFFU, \
  disp1_scalar_lolo & 0xFFFFFFFFU, disp1_scalar_lohi & 0xFFFFFFFFU, disp1_scalar_hilo & 0xFFFFFFFFU, disp1_scalar_hihi & 0xFFFFFFFFU, \
  disp2_scalar_lolo & 0xFFFFFFFFU, disp2_scalar_lohi & 0xFFFFFFFFU, disp2_scalar_hilo & 0xFFFFFFFFU, disp2_scalar_hihi & 0xFFFFFFFFU
			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(0, 0, 0, 0));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(0, 0, 0, 0));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(0, 0, 0, 0));
			idx_scalar_lolo = EXTRACT_EVEN(idx_scalar_lolo, idx_lolo, 0);
			idx_scalar_lohi = EXTRACT_EVEN(idx_scalar_lohi, idx_lohi, 0);
			idx_scalar_hilo = EXTRACT_EVEN(idx_scalar_hilo, idx_hilo, 0);
			idx_scalar_hihi = EXTRACT_EVEN(idx_scalar_hihi, idx_hihi, 0);
			diag_scalar_lolo = EXTRACT_EVEN(diag_scalar_lolo, diag_lolo, 0);
			diag_scalar_lohi = EXTRACT_EVEN(diag_scalar_lohi, diag_lohi, 0);
			diag_scalar_hilo = EXTRACT_EVEN(diag_scalar_hilo, diag_hilo, 0);
			diag_scalar_hihi = EXTRACT_EVEN(diag_scalar_hihi, diag_hihi, 0);
			disp1_scalar_lolo = EXTRACT_EVEN(disp1_scalar_lolo, disp1_lolo, 0);
			disp1_scalar_lohi = EXTRACT_EVEN(disp1_scalar_lohi, disp1_lohi, 0);
			disp1_scalar_hilo = EXTRACT_EVEN(disp1_scalar_hilo, disp1_hilo, 0);
			disp1_scalar_hihi = EXTRACT_EVEN(disp1_scalar_hihi, disp1_hihi, 0);
			disp2_scalar_lolo = EXTRACT_EVEN(disp2_scalar_lolo, disp2_lolo, 0);
			disp2_scalar_lohi = EXTRACT_EVEN(disp2_scalar_lohi, disp2_lohi, 0);
			disp2_scalar_hilo = EXTRACT_EVEN(disp2_scalar_hilo, disp2_hilo, 0);
			disp2_scalar_hihi = EXTRACT_EVEN(disp2_scalar_hihi, disp2_hihi, 0);
			result048c = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(1, 1, 1, 1));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(1, 1, 1, 1));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(1, 1, 1, 1));
			idx_scalar_lolo = EXTRACT_ODD(idx_scalar_lolo, idx_lolo, 1);
			idx_scalar_lohi = EXTRACT_ODD(idx_scalar_lohi, idx_lohi, 1);
			idx_scalar_hilo = EXTRACT_ODD(idx_scalar_hilo, idx_hilo, 1);
			idx_scalar_hihi = EXTRACT_ODD(idx_scalar_hihi, idx_hihi, 1);
			diag_scalar_lolo = EXTRACT_ODD(diag_scalar_lolo, diag_lolo, 1);
			diag_scalar_lohi = EXTRACT_ODD(diag_scalar_lohi, diag_lohi, 1);
			diag_scalar_hilo = EXTRACT_ODD(diag_scalar_hilo, diag_hilo, 1);
			diag_scalar_hihi = EXTRACT_ODD(diag_scalar_hihi, diag_hihi, 1);
			disp1_scalar_lolo = EXTRACT_ODD(disp1_scalar_lolo, disp1_lolo, 1);
			disp1_scalar_lohi = EXTRACT_ODD(disp1_scalar_lohi, disp1_lohi, 1);
			disp1_scalar_hilo = EXTRACT_ODD(disp1_scalar_hilo, disp1_hilo, 1);
			disp1_scalar_hihi = EXTRACT_ODD(disp1_scalar_hihi, disp1_hihi, 1);
			disp2_scalar_lolo = EXTRACT_ODD(disp2_scalar_lolo, disp2_lolo, 1);
			disp2_scalar_lohi = EXTRACT_ODD(disp2_scalar_lohi, disp2_lohi, 1);
			disp2_scalar_hilo = EXTRACT_ODD(disp2_scalar_hilo, disp2_hilo, 1);
			disp2_scalar_hihi = EXTRACT_ODD(disp2_scalar_hihi, disp2_hihi, 1);
			result159d = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(2, 2, 2, 2));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(2, 2, 2, 2));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(2, 2, 2, 2));
			idx_scalar_lolo = EXTRACT_EVEN(idx_scalar_lolo, idx_lolo, 2);
			idx_scalar_lohi = EXTRACT_EVEN(idx_scalar_lohi, idx_lohi, 2);
			idx_scalar_hilo = EXTRACT_EVEN(idx_scalar_hilo, idx_hilo, 2);
			idx_scalar_hihi = EXTRACT_EVEN(idx_scalar_hihi, idx_hihi, 2);
			diag_scalar_lolo = EXTRACT_EVEN(diag_scalar_lolo, diag_lolo, 2);
			diag_scalar_lohi = EXTRACT_EVEN(diag_scalar_lohi, diag_lohi, 2);
			diag_scalar_hilo = EXTRACT_EVEN(diag_scalar_hilo, diag_hilo, 2);
			diag_scalar_hihi = EXTRACT_EVEN(diag_scalar_hihi, diag_hihi, 2);
			disp1_scalar_lolo = EXTRACT_EVEN(disp1_scalar_lolo, disp1_lolo, 2);
			disp1_scalar_lohi = EXTRACT_EVEN(disp1_scalar_lohi, disp1_lohi, 2);
			disp1_scalar_hilo = EXTRACT_EVEN(disp1_scalar_hilo, disp1_hilo, 2);
			disp1_scalar_hihi = EXTRACT_EVEN(disp1_scalar_hihi, disp1_hihi, 2);
			disp2_scalar_lolo = EXTRACT_EVEN(disp2_scalar_lolo, disp2_lolo, 2);
			disp2_scalar_lohi = EXTRACT_EVEN(disp2_scalar_lohi, disp2_lohi, 2);
			disp2_scalar_hilo = EXTRACT_EVEN(disp2_scalar_hilo, disp2_hilo, 2);
			disp2_scalar_hihi = EXTRACT_EVEN(disp2_scalar_hihi, disp2_hihi, 2);
			result26ae = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm512_permute_ps(r, _MM_SHUFFLE(3, 3, 3, 3));
			gtmp = _mm512_permute_ps(g, _MM_SHUFFLE(3, 3, 3, 3));
			btmp = _mm512_permute_ps(b, _MM_SHUFFLE(3, 3, 3, 3));
			idx_scalar_lolo = EXTRACT_ODD(idx_scalar_lolo, idx_lolo, 3);
			idx_scalar_lohi = EXTRACT_ODD(idx_scalar_lohi, idx_lohi, 3);
			idx_scalar_hilo = EXTRACT_ODD(idx_scalar_hilo, idx_hilo, 3);
			idx_scalar_hihi = EXTRACT_ODD(idx_scalar_hihi, idx_hihi, 3);
			diag_scalar_lolo = EXTRACT_ODD(diag_scalar_lolo, diag_lolo, 3);
			diag_scalar_lohi = EXTRACT_ODD(diag_scalar_lohi, diag_lohi, 3);
			diag_scalar_hilo = EXTRACT_ODD(diag_scalar_hilo, diag_hilo, 3);
			diag_scalar_hihi = EXTRACT_ODD(diag_scalar_hihi, diag_hihi, 3);
			disp1_scalar_lolo = EXTRACT_ODD(disp1_scalar_lolo, disp1_lolo, 3);
			disp1_scalar_lohi = EXTRACT_ODD(disp1_scalar_lohi, disp1_lohi, 3);
			disp1_scalar_hilo = EXTRACT_ODD(disp1_scalar_hilo, disp1_hilo, 3);
			disp1_scalar_hihi = EXTRACT_ODD(disp1_scalar_hihi, disp1_hihi, 3);
			disp2_scalar_lolo = EXTRACT_ODD(disp2_scalar_lolo, disp2_lolo, 3);
			disp2_scalar_lohi = EXTRACT_ODD(disp2_scalar_lohi, disp2_lohi, 3);
			disp2_scalar_hilo = EXTRACT_ODD(disp2_scalar_hilo, disp2_hilo, 3);
			disp2_scalar_hihi = EXTRACT_ODD(disp2_scalar_hihi, disp2_hihi, 3);
			result37bf = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);
#undef INDICES
#undef EXTRACT_ODD
#undef EXTRACT_EVEN
			lut3d_unpack_result(result048c, result159d, result26ae, result37bf, r, g, b);

			_mm512_store_ps(dst_r + i, r);
			_mm512_store_ps(dst_g + i, g);
			_mm512_store_ps(dst_b + i, b);
		}
	}
};


} // namespace


void byte_to_float_avx512(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned)
{
	const uint8_t *srcp = static_cast<const uint8_t *>(src);
	float *dstp = static_cast<float *>(dst);
	const __m512 scale_ps = _mm512_set1_ps(scale);
	const __m512 offset_ps = _mm512_set1_ps(offset);

	for (unsigned i = left; i < right; i += 16) {
		__m512i x = _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(srcp + i)));
		__m512 y = _mm512_cvtepi32_ps(x);

		y = _mm512_fmadd_ps(scale_ps, y, offset_ps);
		_mm512_store_ps(dstp + i, y);
	}
}

void word_to_float_avx512(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned)
{
	const uint16_t *srcp = static_cast<const uint16_t *>(src);
	float *dstp = static_cast<float *>(dst);
	const __m512 scale_ps = _mm512_set1_ps(scale);
	const __m512 offset_ps = _mm512_set1_ps(offset);

	for (unsigned i = left; i < right; i += 16) {
		__m512i x = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i *)(srcp + i)));
		__m512 y = _mm512_cvtepi32_ps(x);

		y = _mm512_fmadd_ps(scale_ps, y, offset_ps);
		_mm512_store_ps(dstp + i, y);
	}
}

void half_to_float_avx512(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned)
{
	const uint16_t *srcp = static_cast<const uint16_t *>(src);
	float *dstp = static_cast<float *>(dst);

	for (unsigned i = left; i < right; i += 16) {
		__m256i x = _mm256_load_si256((const __m256i *)(srcp + i));
		__m512 y = _mm512_cvtph_ps(x);
		_mm512_store_ps(dstp + i, y);
	}
}

void float_to_byte_avx512(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned depth)
{
	const float *srcp = static_cast<const float *>(src);
	uint8_t *dstp = static_cast<uint8_t *>(dst);
	const __m512 scale_ps = _mm512_set1_ps(scale);
	const __m512 offset_ps = _mm512_set1_ps(offset);
	const __m128i maxval = _mm_set1_epi8((1U << depth) - 1);

	for (unsigned i = left; i < right; i += 16) {
		__m512 x;
		__m512i xi;

		x = _mm512_load_ps(srcp + i);
		x = _mm512_fmadd_ps(scale_ps, x, offset_ps);
		xi = _mm512_cvtps_epi32(x);
		_mm_store_si128((__m128i *)(dstp + i), _mm_min_epu8(_mm512_cvtusepi32_epi8(xi), maxval));
	}
}

void float_to_word_avx512(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned depth)
{
	const float *srcp = static_cast<const float *>(src);
	uint16_t *dstp = static_cast<uint16_t *>(dst);
	const __m512 scale_ps = _mm512_set1_ps(scale);
	const __m512 offset_ps = _mm512_set1_ps(offset);
	const __m256i maxval = _mm256_set1_epi8((1U << depth) - 1);

	for (unsigned i = left; i < right; i += 16) {
		__m512 x;
		__m512i xi;

		x = _mm512_load_ps(srcp + i);
		x = _mm512_fmadd_ps(scale_ps, x, offset_ps);
		xi = _mm512_cvtps_epi32(x);
		_mm256_store_si256((__m256i *)(dstp + i), _mm256_min_epu16(_mm512_cvtusepi32_epi16(xi), maxval));
	}
}

void float_to_half_avx512(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned)
{
	const float *srcp = static_cast<const float *>(src);
	uint16_t *dstp = static_cast<uint16_t *>(dst);

	for (unsigned i = left; i < right; i += 16) {
		__m512 x = _mm512_load_ps(srcp + i);
		__m256i y = _mm512_cvtps_ph(x, 0);
		_mm256_store_si256((__m256i *)(dstp + i), y);
	}
}


std::unique_ptr<graphengine::Filter> create_lut3d_impl_avx512(const Cube &cube, unsigned width, unsigned height, Interpolation interp)
{
	if (interp == Interpolation::TETRA)
		return std::make_unique<TetrahedralFilter_AVX512>(cube, width, height);
	else
		return std::make_unique<TrilinearFilter_AVX512>(cube, width, height);
}

} // namespace timecube

#endif // CUBE_X86
