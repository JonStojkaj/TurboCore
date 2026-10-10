#include "detail/cpu_features.hpp"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define TURBOCORE_ARCH_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define TURBOCORE_ARCH_X86 0
#endif

namespace turbocore::detail {

#if TURBOCORE_ARCH_X86

namespace {

struct CpuidRegs {
  std::uint32_t eax, ebx, ecx, edx;
};

CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  return {static_cast<std::uint32_t>(regs[0]), static_cast<std::uint32_t>(regs[1]),
          static_cast<std::uint32_t>(regs[2]), static_cast<std::uint32_t>(regs[3])};
#else
  CpuidRegs regs{};
  __cpuid_count(leaf, subleaf, regs.eax, regs.ebx, regs.ecx, regs.edx);
  return regs;
#endif
}

std::uint64_t xgetbv0() noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
  return _xgetbv(0);
#else
  // Raw opcode instead of _xgetbv() so this file needs no -mxsave.
  std::uint32_t eax = 0;
  std::uint32_t edx = 0;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

constexpr bool bit(std::uint32_t reg, unsigned index) noexcept { return ((reg >> index) & 1u) != 0; }

}  // namespace

bool cpu_supports_avx2_backend() noexcept {
  const std::uint32_t max_leaf = cpuid(0, 0).eax;
  if (max_leaf < 7) {
    return false;
  }

  const CpuidRegs leaf1 = cpuid(1, 0);
  const bool sse42 = bit(leaf1.ecx, 20);  // CRC32 instruction
  const bool osxsave = bit(leaf1.ecx, 27);
  const bool avx = bit(leaf1.ecx, 28);
  const bool f16c = bit(leaf1.ecx, 29);
  if (!sse42 || !osxsave || !avx || !f16c) {
    return false;
  }
  // XCR0 bits 1 and 2: the OS saves and restores XMM and YMM state.
  if ((xgetbv0() & 0x6u) != 0x6u) {
    return false;
  }

  const CpuidRegs leaf7 = cpuid(7, 0);
  const bool bmi1 = bit(leaf7.ebx, 3);
  const bool avx2 = bit(leaf7.ebx, 5);
  const bool bmi2 = bit(leaf7.ebx, 8);

  const bool lzcnt = cpuid(0x80000000u, 0).eax >= 0x80000001u && bit(cpuid(0x80000001u, 0).ecx, 5);

  return avx2 && bmi1 && bmi2 && lzcnt;
}

#else

bool cpu_supports_avx2_backend() noexcept { return false; }

#endif

}  // namespace turbocore::detail
