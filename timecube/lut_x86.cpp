#ifdef CUBE_X86

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__)
  #include <cpuid.h>
#endif

#include "cube.h"
#include "lut.h"
#include "lut_x86.h"


namespace timecube {
namespace {

/**
 * Bitfield of selected x86 feature flags.
 */
struct X86Capabilities {
	unsigned sse   : 1;
	unsigned sse2  : 1;
	unsigned sse3  : 1;
	unsigned ssse3 : 1;
	unsigned fma   : 1;
	unsigned sse41 : 1;
	unsigned sse42 : 1;
	unsigned avx   : 1;
	unsigned f16c  : 1;
	unsigned avx2  : 1;
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

	do_cpuid(regs, 1, 0);
	caps.sse   = !!(regs[3] & (1 << 25));
	caps.sse2  = !!(regs[3] & (1 << 26));
	caps.sse3  = !!(regs[2] & (1 << 0));
	caps.ssse3 = !!(regs[2] & (1 << 9));
	caps.fma   = !!(regs[2] & (1 << 12));
	caps.sse41 = !!(regs[2] & (1 << 19));
	caps.sse42 = !!(regs[2] & (1 << 20));

	// osxsave
	if (regs[2] & (1 << 27))
		xcr0 = do_xgetbv(0);

	// XMM and YMM state.
	if ((xcr0 & 0x06) != 0x06)
		return caps;

	caps.avx   = !!(regs[2] & (1 << 28));
	caps.f16c  = !!(regs[2] & (1 << 29));

	do_cpuid(regs, 7, 0);
	caps.avx2  = !!(regs[1] & (1 << 5));

	return caps;
}

} // namespace


std::unique_ptr<Lut> create_lut_impl_x86(const Cube &cube)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Lut> ret;

	if (!ret && caps.avx2 && caps.fma)
		ret = create_lut_impl_avx2(cube);
	if (!ret && caps.sse41)
		ret = create_lut_impl_sse41(cube);

	return ret;
}

} // namespace timecube

#endif // CUBE_X86
