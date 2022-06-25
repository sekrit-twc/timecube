#ifdef CUBE_X86

#include <climits>

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__)
  #include <cpuid.h>
#endif

#include <graphengine/filter.h>
#include "cube.h"
#include "lut.h"
#include "lut_x86.h"


namespace timecube {
namespace {

enum {
	SIMD_NONE   = 0,
	SIMD_SSE42  = 1,
	SIMD_AVX2   = 2,
	SIMD_AVX512 = 3,
	SIMD_MAX    = INT_MAX,
};

/**
 * Bitfield of selected x86 feature flags.
 */
struct X86Capabilities {
	unsigned sse                : 1;
	unsigned sse2               : 1;
	unsigned sse3               : 1;
	unsigned ssse3              : 1;
	unsigned fma                : 1;
	unsigned sse41              : 1;
	unsigned sse42              : 1;
	unsigned avx                : 1;
	unsigned f16c               : 1;
	unsigned avx2               : 1;
	unsigned avxvnni            : 1;
	unsigned avx512f            : 1;
	unsigned avx512dq           : 1;
	unsigned avx512ifma         : 1;
	unsigned avx512cd           : 1;
	unsigned avx512bw           : 1;
	unsigned avx512vl           : 1;
	unsigned avx512vbmi         : 1;
	unsigned avx512vbmi2        : 1;
	unsigned avx512vnni         : 1;
	unsigned avx512bitalg       : 1;
	unsigned avx512vpopcntdq    : 1;
	unsigned avx512vp2intersect : 1;
	unsigned avx512fp16         : 1;
	unsigned avx512bf16         : 1;
};
/**
 * Execute the CPUID instruction.
 *
 * @param regs array to receive eax, ebx, ecx, edx
 * @param eax argument to instruction
 * @param ecx argument to instruction
 */
void do_cpuid(int regs[4], int eax, int ecx)
{
#if defined(_MSC_VER)
	__cpuidex(regs, eax, ecx);
#elif defined(__GNUC__)
	__cpuid_count(eax, ecx, regs[0], regs[1], regs[2], regs[3]);
#else
	regs[0] = 0;
	regs[1] = 0;
	regs[2] = 0;
	regs[3] = 0;
#endif
}


/**
 * Execute the XGETBV instruction.
 *
 * @param ecx argument to instruction
 * @return (edx << 32) | eax
 */
unsigned long long do_xgetbv(unsigned ecx)
{
#if defined(_MSC_VER)
	return _xgetbv(ecx);
#elif defined(__GNUC__)
	unsigned eax, edx;
	__asm("xgetbv" : "=a"(eax), "=d"(edx) : "c"(ecx) : );
	return (static_cast<unsigned long long>(edx) << 32) | eax;
#else
	return 0;
#endif
}

X86Capabilities query_x86_capabilities() noexcept
{
	X86Capabilities caps = { 0 };
	unsigned long long xcr0 = 0;
	int regs[4] = { 0 };
	int xmmymm = 0;
	int zmm = 0;

	do_cpuid(regs, 1, 0);
	caps.sse      = !!(regs[3] & (1U << 25));
	caps.sse2     = !!(regs[3] & (1U << 26));
	caps.sse3     = !!(regs[2] & (1U << 0));
	caps.ssse3    = !!(regs[2] & (1U << 9));
	caps.fma      = !!(regs[2] & (1U << 12));
	caps.sse41    = !!(regs[2] & (1U << 19));
	caps.sse42    = !!(regs[2] & (1U << 20));

	// osxsave
	if (regs[2] & (1U << 27)) {
		xcr0 = do_xgetbv(0);
		xmmymm = (xcr0 & 0x06) == 0x06;
		zmm = (xcr0 & 0xE0) == 0xE0;
	}

	// XMM and YMM state.
	if (xmmymm) {
		caps.avx  = !!(regs[2] & (1U << 28));
		caps.f16c = !!(regs[2] & (1U << 29));
	}

	do_cpuid(regs, 7, 0);
	if (xmmymm) {
		caps.avx2 = !!(regs[1] & (1U << 5));
	}

	// ZMM state.
	if (zmm) {
		caps.avx512f            = !!(regs[1] & (1U << 16));
		caps.avx512dq           = !!(regs[1] & (1U << 17));
		caps.avx512ifma         = !!(regs[1] & (1U << 21));
		caps.avx512cd           = !!(regs[1] & (1U << 28));
		caps.avx512bw           = !!(regs[1] & (1U << 30));
		caps.avx512vl           = !!(regs[1] & (1U << 31));
		caps.avx512vbmi         = !!(regs[2] & (1U << 1));
		caps.avx512vbmi2        = !!(regs[2] & (1U << 6));
		caps.avx512vnni         = !!(regs[2] & (1U << 11));
		caps.avx512bitalg       = !!(regs[2] & (1U << 12));
		caps.avx512vpopcntdq    = !!(regs[2] & (1U << 14));
		caps.avx512vp2intersect = !!(regs[3] & (1U << 8));
		caps.avx512fp16         = !!(regs[3] & (1U << 23));
	}

	do_cpuid(regs, 7, 1);
	if (zmm) {
		caps.avxvnni            = !!(regs[0] & (1U << 4));
		caps.avx512bf16         = !!(regs[0] & (1U << 5));
	}

	return caps;
}


pixel_io_func select_from_float_func_sse41(PixelType to)
{
	switch (to) {
	case PixelType::BYTE:
		return float_to_byte_sse41;
	case PixelType::WORD:
		return float_to_word_sse41;
	default:
		return nullptr;
	}
}

pixel_io_func select_from_float_func_avx2(PixelType to)
{
	switch (to) {
	case PixelType::BYTE:
		return float_to_byte_avx2;
	case PixelType::WORD:
		return float_to_word_avx2;
	case PixelType::HALF:
		return float_to_half_avx2;
	default:
		return nullptr;
	}
}

pixel_io_func select_from_float_func_avx512(PixelType to)
{
	switch (to) {
	case PixelType::BYTE:
		return float_to_byte_avx512;
	case PixelType::WORD:
		return float_to_word_avx512;
	case PixelType::HALF:
		return float_to_half_avx512;
	default:
		return nullptr;
	}
}

pixel_io_func select_to_float_func_sse41(PixelType from)
{
	switch (from) {
	case PixelType::BYTE:
		return byte_to_float_sse41;
	case PixelType::WORD:
		return word_to_float_sse41;
	default:
		return nullptr;
	}
}

pixel_io_func select_to_float_func_avx2(PixelType from)
{
	switch (from) {
	case PixelType::BYTE:
		return byte_to_float_avx2;
	case PixelType::WORD:
		return word_to_float_avx2;
	case PixelType::HALF:
		return half_to_float_avx2;
	default:
		return nullptr;
	}
}

pixel_io_func select_to_float_func_avx512(PixelType from)
{
	switch (from) {
	case PixelType::BYTE:
		return byte_to_float_avx512;
	case PixelType::WORD:
		return word_to_float_avx512;
	case PixelType::HALF:
		return half_to_float_avx512;
	default:
		return nullptr;
	}
}

} // namespace


std::unique_ptr<graphengine::Filter> create_lut3d_impl_x86(const Cube &cube, unsigned width, unsigned height, int simd)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (!ret && simd >= SIMD_AVX512 && caps.avx512f && caps.avx512bw && caps.avx512dq && caps.avx512vl)
		ret = create_lut3d_impl_avx512(cube, width, height);
	if (!ret && simd >= SIMD_AVX2 && caps.avx2 && caps.fma)
		ret = create_lut3d_impl_avx2(cube, width, height);
	if (!ret && simd >= SIMD_SSE42 && caps.sse41)
		ret = create_lut3d_impl_sse41(cube, width, height);

	return ret;
}

pixel_io_func select_from_float_func_x86(PixelType to, int simd)
{
	X86Capabilities caps = query_x86_capabilities();
	pixel_io_func ret = nullptr;

	if (!ret && simd >= SIMD_AVX512 && caps.avx512f && caps.avx512bw && caps.avx512dq && caps.avx512vl)
		ret = select_from_float_func_avx512(to);
	if (!ret && simd >= SIMD_AVX2 && caps.avx2 && caps.fma)
		ret = select_from_float_func_avx2(to);
	if (!ret && simd >= SIMD_SSE42 && caps.sse41)
		ret = select_from_float_func_sse41(to);

	return ret;
}

pixel_io_func select_to_float_func_x86(PixelType from, int simd)
{
	X86Capabilities caps = query_x86_capabilities();
	pixel_io_func ret = nullptr;

	if (!ret && simd >= SIMD_AVX512 && caps.avx512f && caps.avx512bw && caps.avx512dq && caps.avx512vl)
		ret = select_to_float_func_avx512(from);
	if (!ret && simd >= SIMD_AVX2 && caps.avx2 && caps.fma)
		ret = select_to_float_func_avx2(from);
	if (!ret && simd >= SIMD_SSE42 && caps.sse41)
		ret = select_to_float_func_sse41(from);

	return ret;
}

} // namespace timecube

#endif // CUBE_X86
