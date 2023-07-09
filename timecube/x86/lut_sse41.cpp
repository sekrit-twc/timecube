#ifdef CUBE_X86

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <smmintrin.h>
#include <graphengine/filter.h>
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

	// (1 - x) * a == a - x * a
	x = _mm_mul_ps(dist, lo);
	x = _mm_sub_ps(lo, x);
	// (a - x * a) + x * b
	x = _mm_add_ps(_mm_mul_ps(dist, hi), x);

	return x;
}

static inline FORCE_INLINE __m128i lut3d_calculate_index(const __m128 &r, const __m128 &g, const __m128 &b, ptrdiff_t stride_g, ptrdiff_t stride_b)
{
	__m128i idx_r, idx_g, idx_b;

	idx_r = _mm_cvttps_epi32(r);
	idx_r = _mm_slli_epi32(idx_r, 4); // 16 byte entries.

	idx_g = _mm_cvttps_epi32(g);
	idx_g = _mm_mullo_epi32(idx_g, _mm_set1_epi32(static_cast<uint32_t>(stride_g)));

	idx_b = _mm_cvttps_epi32(b);
	idx_b = _mm_mullo_epi32(idx_b, _mm_set1_epi32(static_cast<uint32_t>(stride_b)));

	return _mm_add_epi32(_mm_add_epi32(idx_r, idx_g), idx_b);
}

// Performs trilinear interpolation on one pixel.
// Returns [R G B x].
static inline FORCE_INLINE __m128 lut3d_trilinear_interp(const void *lut, ptrdiff_t stride_g, ptrdiff_t stride_b, ptrdiff_t idx,
                                                         __m128 r, __m128 g, __m128 b)
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

static inline FORCE_INLINE void minmax(__m128 &x, __m128 &y)
{
	__m128 minval = _mm_min_ps(x, y);
	__m128 maxval = _mm_max_ps(x, y);
	x = minval;
	y = maxval;
}

// Identify sub-tetra containing each pixel.
// Writes-back normalized (sorted) pixel coordinates into r, g, b.
// Returns absolute displacements from vertexes (0,0,0) and (1,1,1) in disp1, disp2.
static inline FORCE_INLINE void lut3d_tetra_classify(__m128 &r, __m128 &g, __m128 &b, __m128i &disp1, __m128i &disp2,
                                                     ptrdiff_t stride_g, ptrdiff_t stride_b)
{
	__m128 x = r, y = g, z = b;
	__m128i stride_r_epi32 = _mm_set1_epi32(sizeof(float) * 4);
	__m128i stride_g_epi32 = _mm_set1_epi32(static_cast<uint32_t>(stride_g));
	__m128i stride_b_epi32 = _mm_set1_epi32(static_cast<uint32_t>(stride_b));
	__m128i tmp;

	// Sort.
	minmax(x, z);
	minmax(x, y);
	minmax(y, z);

	tmp = stride_b_epi32;
	tmp = _mm_blendv_epi8(tmp, stride_g_epi32, _mm_cmpeq_epi32(_mm_castps_si128(z), _mm_castps_si128(g)));
	tmp = _mm_blendv_epi8(tmp, stride_r_epi32, _mm_cmpeq_epi32(_mm_castps_si128(z), _mm_castps_si128(r)));
	disp1 = tmp;

	tmp = stride_b_epi32;
	tmp = _mm_blendv_epi8(tmp, stride_g_epi32, _mm_cmpeq_epi32(_mm_castps_si128(x), _mm_castps_si128(g)));
	tmp = _mm_blendv_epi8(tmp, stride_r_epi32, _mm_cmpeq_epi32(_mm_castps_si128(x), _mm_castps_si128(r)));
	disp2 = tmp;

	r = x;
	g = y;
	b = z;
}

// Performs tetrahedral interpolation on one pixel.
// Returns [R G B x]
static inline FORCE_INLINE __m128 lut3d_tetra_interp(const void *lut, ptrdiff_t idx_base, ptrdiff_t idx_diag, ptrdiff_t idx_disp1, ptrdiff_t idx_disp2,
                                                     __m128 x, __m128 y, __m128 z)
{
#define LUT_OFFSET(x) reinterpret_cast<const float *>(static_cast<const unsigned char *>(lut) + (x))
	__m128 v0 = _mm_load_ps(LUT_OFFSET(idx_base));
	__m128 v1 = _mm_load_ps(LUT_OFFSET(idx_diag));
	__m128 v2 = _mm_load_ps(LUT_OFFSET(idx_disp1));
	__m128 v3 = _mm_load_ps(LUT_OFFSET(idx_disp2));

	// (1 - z) * v0 + x * v1 + (z - y) * v2 + (y - x) * v3
	//   ==>
	// v0 - z * v0 + x * v1 + z * v2 - y * v2 + y * v3 - x * v3
	__m128 result = v0;
	result = _mm_sub_ps(result, _mm_mul_ps(z, v0));
	result = _mm_add_ps(result, _mm_mul_ps(x, v1));
	result = _mm_add_ps(result, _mm_mul_ps(z, v2));
	result = _mm_sub_ps(result, _mm_mul_ps(y, v2));
	result = _mm_add_ps(result, _mm_mul_ps(y, v3));
	result = _mm_sub_ps(result, _mm_mul_ps(x, v3));
	return result;
#undef LUT_OFFSET
}


class Lut3DFilter_SSE41 : public Lut3DFilter {
protected:
	std::vector<float, AlignedAllocator<float>> m_lut;
	uint32_t m_dim;
	float m_scale[3];
	float m_offset[3];

	Lut3DFilter_SSE41(const Cube &cube, unsigned width, unsigned height) :
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

		m_desc.alignment_mask = 0x3;
	}
};

class TrilinearFilter_SSE41 final : public Lut3DFilter_SSE41 {
public:
	TrilinearFilter_SSE41(const Cube &cube, unsigned width, unsigned height) : Lut3DFilter_SSE41(cube, width, height) {}

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

		const __m128 scale_r = _mm_set_ps1(m_scale[0]);
		const __m128 scale_g = _mm_set_ps1(m_scale[1]);
		const __m128 scale_b = _mm_set_ps1(m_scale[2]);
		const __m128 offset_r = _mm_set_ps1(m_offset[0]);
		const __m128 offset_g = _mm_set_ps1(m_offset[1]);
		const __m128 offset_b = _mm_set_ps1(m_offset[2]);

		const __m128 lut_max = _mm_set_ps1(std::nextafter(static_cast<float>(m_dim - 1), -INFINITY));

		for (unsigned i = left; i < right; i += 4) {
			__m128 r = _mm_load_ps(src_r + i);
			__m128 g = _mm_load_ps(src_g + i);
			__m128 b = _mm_load_ps(src_b + i);

			__m128 result0, result1, result2, result3;
			__m128 rtmp, gtmp, btmp;
			__m128i idx;

			size_t idx_scalar;

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

			idx = lut3d_calculate_index(r, g, b, lut_stride_g, lut_stride_b);

			// Cube distances.
			r = _mm_sub_ps(r, _mm_floor_ps(r));
			g = _mm_sub_ps(g, _mm_floor_ps(g));
			b = _mm_sub_ps(b, _mm_floor_ps(b));

			// Interpolation.
#if SIZE_MAX >= UINT64_MAX
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi64((x), (idx) / 2)
  #define EXTRACT_ODD(out, x, idx) ((out) >> 32)
#else
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi32((x), (idx))
  #define EXTRACT_ODD(out, x, idx) _mm_extract_epi32((x), (idx))
#endif
			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 0));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(0, 0, 0, 0));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 0, 0));
			idx_scalar = EXTRACT_EVEN(idx_scalar, idx, 0);
			result0 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar & 0xFFFFFFFFU, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 1, 1, 1));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(1, 1, 1, 1));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1));
			idx_scalar = EXTRACT_ODD(idx_scalar, idx, 1);
			result1 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 2, 2, 2));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(2, 2, 2, 2));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 2, 2));
			idx_scalar = EXTRACT_EVEN(idx_scalar, idx, 2);
			result2 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar & 0xFFFFFFFFU, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 3, 3, 3));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(3, 3, 3, 3));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3));
			idx_scalar = EXTRACT_ODD(idx_scalar, idx, 3);
			result3 = lut3d_trilinear_interp(lut, lut_stride_g, lut_stride_b, idx_scalar, rtmp, gtmp, btmp);
#undef EXTRACT_ODD
#undef EXTRACT_EVEN

			_MM_TRANSPOSE4_PS(result0, result1, result2, result3);
			r = result0;
			g = result1;
			b = result2;

			_mm_store_ps(dst_r + i, r);
			_mm_store_ps(dst_g + i, g);
			_mm_store_ps(dst_b + i, b);
		}
	}
};

class TetrahedralFilter_SSE41 final : public Lut3DFilter_SSE41 {
public:
	TetrahedralFilter_SSE41(const Cube &cube, unsigned width, unsigned height) : Lut3DFilter_SSE41(cube, width, height) {}

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

		const __m128 scale_r = _mm_set_ps1(m_scale[0]);
		const __m128 scale_g = _mm_set_ps1(m_scale[1]);
		const __m128 scale_b = _mm_set_ps1(m_scale[2]);
		const __m128 offset_r = _mm_set_ps1(m_offset[0]);
		const __m128 offset_g = _mm_set_ps1(m_offset[1]);
		const __m128 offset_b = _mm_set_ps1(m_offset[2]);

		const __m128 lut_max = _mm_set_ps1(std::nextafter(static_cast<float>(m_dim - 1), -INFINITY));

		for (unsigned i = left; i < right; i += 4) {
			__m128 r = _mm_load_ps(src_r + i);
			__m128 g = _mm_load_ps(src_g + i);
			__m128 b = _mm_load_ps(src_b + i);

			__m128 result0, result1, result2, result3;
			__m128 rtmp, gtmp, btmp;
			__m128i idx, diag, disp1, disp2;

			size_t idx_scalar, diag_scalar, disp1_scalar, disp2_scalar;

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

			idx = lut3d_calculate_index(r, g, b, lut_stride_g, lut_stride_b);
			diag = _mm_add_epi32(idx, _mm_set1_epi32(lut_stride_diag));

			// Cube distances.
			r = _mm_sub_ps(r, _mm_floor_ps(r));
			g = _mm_sub_ps(g, _mm_floor_ps(g));
			b = _mm_sub_ps(b, _mm_floor_ps(b));

			// Classification.
			lut3d_tetra_classify(r, g, b, disp1, disp2, lut_stride_g, lut_stride_b);
			disp1 = _mm_add_epi32(idx, disp1);
			disp2 = _mm_sub_epi32(diag, disp2);

			// Interpolation.
#if SIZE_MAX >= UINT64_MAX
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi64((x), (idx) / 2)
  #define EXTRACT_ODD(out, x, idx) ((out) >> 32)
#else
  #define EXTRACT_EVEN(out, x, idx) _mm_extract_epi32((x), (idx))
  #define EXTRACT_ODD(out, x, idx) _mm_extract_epi32((x), (idx))
#endif

#define INDICES idx_scalar & 0xFFFFFFFFU, diag_scalar & 0xFFFFFFFFU, disp1_scalar & 0xFFFFFFFFU, disp2_scalar & 0xFFFFFFFFU
			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 0));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(0, 0, 0, 0));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 0, 0));
			idx_scalar = EXTRACT_EVEN(idx_scalar, idx, 0);
			diag_scalar = EXTRACT_EVEN(diag_scalar, diag, 0);
			disp1_scalar = EXTRACT_EVEN(disp1_scalar, disp1, 0);
			disp2_scalar = EXTRACT_EVEN(disp2_scalar, disp2, 0);
			result0 = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 1, 1, 1));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(1, 1, 1, 1));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1));
			idx_scalar = EXTRACT_ODD(idx_scalar, idx, 1);
			diag_scalar = EXTRACT_ODD(diag_scalar, diag, 1);
			disp1_scalar = EXTRACT_ODD(disp1_scalar, disp1, 1);
			disp2_scalar = EXTRACT_ODD(disp2_scalar, disp2, 1);
			result1 = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 2, 2, 2));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(2, 2, 2, 2));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 2, 2));
			idx_scalar = EXTRACT_EVEN(idx_scalar, idx, 2);
			diag_scalar = EXTRACT_EVEN(diag_scalar, diag, 2);
			disp1_scalar = EXTRACT_EVEN(disp1_scalar, disp1, 2);
			disp2_scalar = EXTRACT_EVEN(disp2_scalar, disp2, 2);
			result2 = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);

			rtmp = _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 3, 3, 3));
			gtmp = _mm_shuffle_ps(g, g, _MM_SHUFFLE(3, 3, 3, 3));
			btmp = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3));
			idx_scalar = EXTRACT_ODD(idx_scalar, idx, 3);
			diag_scalar = EXTRACT_ODD(diag_scalar, diag, 3);
			disp1_scalar = EXTRACT_ODD(disp1_scalar, disp1, 3);
			disp2_scalar = EXTRACT_ODD(disp2_scalar, disp2, 3);
			result3 = lut3d_tetra_interp(lut, INDICES, rtmp, gtmp, btmp);
#undef INDICES
#undef EXTRACT_ODD
#undef EXTRACT_EVEN

			_MM_TRANSPOSE4_PS(result0, result1, result2, result3);
			r = result0;
			g = result1;
			b = result2;

			_mm_store_ps(dst_r + i, r);
			_mm_store_ps(dst_g + i, g);
			_mm_store_ps(dst_b + i, b);
		}
	}
};

} // namespace


void byte_to_float_sse41(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned)
{
	const uint8_t *srcp = static_cast<const uint8_t *>(src);
	float *dstp = static_cast<float *>(dst);
	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);

	for (unsigned i = left; i < right; i += 16) {
		__m128i lolo, lohi, hilo, hihi;
		__m128i lo, hi;
		__m128i x;
		__m128 y;

		x = _mm_load_si128((const __m128i *)(srcp + i));

		lo = _mm_unpacklo_epi8(x, _mm_setzero_si128());
		hi = _mm_unpackhi_epi8(x, _mm_setzero_si128());

		lolo = _mm_unpacklo_epi16(lo, _mm_setzero_si128());
		lohi = _mm_unpackhi_epi16(lo, _mm_setzero_si128());
		hilo = _mm_unpacklo_epi16(hi, _mm_setzero_si128());
		hihi = _mm_unpackhi_epi16(hi, _mm_setzero_si128());

		y = _mm_cvtepi32_ps(lolo);
		y = _mm_mul_ps(y, scale_ps);
		y = _mm_add_ps(y, offset_ps);
		_mm_store_ps(dstp + i + 0, y);

		y = _mm_cvtepi32_ps(lohi);
		y = _mm_mul_ps(y, scale_ps);
		y = _mm_add_ps(y, offset_ps);
		_mm_store_ps(dstp + i + 4, y);

		y = _mm_cvtepi32_ps(hilo);
		y = _mm_mul_ps(y, scale_ps);
		y = _mm_add_ps(y, offset_ps);
		_mm_store_ps(dstp + i + 8, y);

		y = _mm_cvtepi32_ps(hihi);
		y = _mm_mul_ps(y, scale_ps);
		y = _mm_add_ps(y, offset_ps);
		_mm_store_ps(dstp + i + 12, y);
	}
}

void word_to_float_sse41(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned)
{
	const uint16_t *srcp = static_cast<const uint16_t *>(src);
	float *dstp = static_cast<float *>(dst);
	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);

	for (unsigned i = left; i < right; i += 8) {
		__m128i lo, hi;
		__m128i x;
		__m128 y;

		x = _mm_load_si128((const __m128i *)(srcp + i));

		lo = _mm_unpacklo_epi16(x, _mm_setzero_si128());
		hi = _mm_unpackhi_epi16(x, _mm_setzero_si128());

		y = _mm_cvtepi32_ps(lo);
		y = _mm_mul_ps(y, scale_ps);
		y = _mm_add_ps(y, offset_ps);
		_mm_store_ps(dstp + i + 0, y);

		y = _mm_cvtepi32_ps(hi);
		y = _mm_mul_ps(y, scale_ps);
		y = _mm_add_ps(y, offset_ps);
		_mm_store_ps(dstp + i + 4, y);
	}
}

void float_to_byte_sse41(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned depth)
{
	const float *srcp = static_cast<const float *>(src);
	uint8_t *dstp = static_cast<uint8_t *>(dst);
	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i maxval = _mm_set1_epi8((1U << depth) - 1);

	for (unsigned i = left; i < right; i += 16) {
		__m128i lolo, lohi, hilo, hihi;
		__m128i lo, hi;
		__m128 x;
		__m128i y;

		x = _mm_load_ps(srcp + i + 0);
		x = _mm_mul_ps(x, scale_ps);
		x = _mm_add_ps(x, offset_ps);
		lolo = _mm_cvtps_epi32(x);

		x = _mm_load_ps(srcp + i + 4);
		x = _mm_mul_ps(x, scale_ps);
		x = _mm_add_ps(x, offset_ps);
		lohi = _mm_cvtps_epi32(x);

		x = _mm_load_ps(srcp + i + 8);
		x = _mm_mul_ps(x, scale_ps);
		x = _mm_add_ps(x, offset_ps);
		hilo = _mm_cvtps_epi32(x);

		x = _mm_load_ps(srcp +i + 12);
		x = _mm_mul_ps(x, scale_ps);
		x = _mm_add_ps(x, offset_ps);
		hihi = _mm_cvtps_epi32(x);

		lo = _mm_packus_epi32(lolo, lohi);
		hi = _mm_packus_epi32(hilo, hihi);

		y = _mm_packus_epi16(lo, hi);
		y = _mm_min_epu8(y, maxval);
		_mm_store_si128((__m128i *)(dstp + i), y);
	}
}

void float_to_word_sse41(const void *src, void *dst, unsigned left, unsigned right, float scale, float offset, unsigned depth)
{
	const float *srcp = static_cast<const float *>(src);
	uint16_t *dstp = static_cast<uint16_t *>(dst);
	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i maxval = _mm_set1_epi8((1U << depth) - 1);

	for (unsigned i = left; i < right; i += 8) {
		__m128i lo, hi;
		__m128 x;
		__m128i y;

		x = _mm_load_ps(srcp + i + 0);
		x = _mm_mul_ps(x, scale_ps);
		x = _mm_add_ps(x, offset_ps);
		lo = _mm_cvtps_epi32(x);

		x = _mm_load_ps(srcp + i + 4);
		x = _mm_mul_ps(x, scale_ps);
		x = _mm_add_ps(x, offset_ps);
		hi = _mm_cvtps_epi32(x);

		y = _mm_packus_epi32(lo, hi);
		y = _mm_min_epu16(y, maxval);
		_mm_store_si128((__m128i *)(dstp + i), y);
	}
}


std::unique_ptr<graphengine::Filter> create_lut3d_impl_sse41(const Cube &cube, unsigned width, unsigned height, Interpolation interp)
{
	if (interp == Interpolation::TETRA)
		return std::make_unique<TetrahedralFilter_SSE41>(cube, width, height);
	else
		return std::make_unique<TrilinearFilter_SSE41>(cube, width, height);
}

} // namespace timecube

#endif // CUBE_X86
