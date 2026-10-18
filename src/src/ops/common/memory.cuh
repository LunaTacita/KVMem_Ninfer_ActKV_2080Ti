#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace ninfer::ops {

// ---------------------------------------------------------------------------------------------
// Alignment self-report, ON while NINFER_ALIGN_DIAG is 1 (set to 0 for a delivery build).
//
// A copy whose two addresses share no alignment is SAFE here -- it falls back to the widest unit
// they do share, down to single bytes -- but it is exactly the condition that faults when a typed
// vector move is used instead, so a run that reports one has found a real contract violation rather
// than a hypothetical. Exactly one thread per kernel reports (the grid-origin thread) and never one
// per element, so a clean run prints nothing and a diagnostic run cannot flood the log.
//
// `smod` is the SHARED-WINDOW offset remainder of the destination and `src`/`srcmod` the source
// address and its remainder. They are printed together on purpose: the destination of every copy on
// this path is shared memory, while the hardware checks the shared offset, so the two agree only
// while the generic shared-window base is itself aligned. Printing both settles that in one run.
// ---------------------------------------------------------------------------------------------
#ifndef NINFER_ALIGN_DIAG
#define NINFER_ALIGN_DIAG 1
#endif

#if NINFER_ALIGN_DIAG
#define NINFER_ALIGN_REPORT(what, need, smod, src, srcmod)                                         \
    do {                                                                                           \
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 &&         \
            blockIdx.y == 0 && blockIdx.z == 0) {                                                  \
            printf("\n[ALIGN] %s need=%u smod=%u src=0x%llx srcmod=%u\n", (what),                  \
                   static_cast<unsigned>(need), static_cast<unsigned>(smod),                       \
                   static_cast<unsigned long long>(src), static_cast<unsigned>(srcmod));           \
        }                                                                                          \
    } while (0)

#define NINFER_VECTOR_ALIGN_CHECK(what, ptr, bytes)                                                \
    do {                                                                                           \
        const std::uintptr_t ninfer_vector_address = reinterpret_cast<std::uintptr_t>(ptr);        \
        if ((ninfer_vector_address & ((bytes) - 1U)) != 0U) {                                      \
            NINFER_ALIGN_REPORT((what), (bytes), ninfer_vector_address & ((bytes) - 1U),           \
                                ninfer_vector_address, ninfer_vector_address & ((bytes) - 1U));   \
        }                                                                                          \
    } while (0)
#else
#define NINFER_ALIGN_REPORT(what, need, smod, src, srcmod) ((void)0)
#define NINFER_VECTOR_ALIGN_CHECK(what, ptr, bytes) ((void)0)
#endif

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    if constexpr (sizeof(V) >= 4) { NINFER_VECTOR_ALIGN_CHECK("load_vec", ptr, sizeof(V)); }
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    if constexpr (sizeof(V) >= 4) { NINFER_VECTOR_ALIGN_CHECK("load_ldg", ptr, sizeof(V)); }
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    if constexpr (sizeof(V) >= 4) { NINFER_VECTOR_ALIGN_CHECK("store_vec", ptr, sizeof(V)); }
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

#if defined(NINFER_SM75)
// ---------------------------------------------------------------------------------------------
// Alignment-safe byte moves.
//
// cp.async moves BYTES and imposes no alignment requirement beyond what the caller's element type
// already guarantees; the int4/int2 typed assignments this port first substituted for it on Turing
// DO carry that type's alignment (16 / 8 bytes), so any caller upstream that passes a row base plus
// a group offset -- 4-byte but not 8-byte aligned for odd rows -- faults with
// cudaErrorMisalignedAddress. That is not hypothetical: the previous sm75 port hit exactly this.
//
// The move is therefore done in the widest unit BOTH addresses satisfy (16, then 8, then 4, then
// single bytes), never narrower than it has to be. Every unit moves the same bytes, so no numerical
// path changes -- but a pair that shares only 4-byte alignment now costs four moves instead of
// sixteen, which is what the earlier all-or-nothing fallback to single bytes gave up.
//
// `dst` is shared memory on every path that reaches here, so its alignment is read from the
// shared-window offset (see copy_bytes) rather than the generic address.
//
// NOTE on diagnosis: a cross-translation-unit counter is NOT available on this build. The tree
// compiles with -rdc=false, where nvcc rejects an inline device variable ("must have internal
// linkage when the program is compiled in whole program mode") and silently treats an `extern`
// device declaration as a static definition (warning #20044-D) -- every translation unit would get
// its own copy, so the readback would always report zero and the counter would be worse than
// useless. Whether a contract is violated is therefore established from behaviour and, while
// NINFER_ALIGN_DIAG is 1, from the [ALIGN] self-report declared at the top of this file.
// ---------------------------------------------------------------------------------------------

template <int Bytes>
__device__ __forceinline__ void copy_bytes(void* dst, const void* src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "copy_bytes supports 4, 8, or 16 bytes");
    auto* d8       = reinterpret_cast<unsigned char*>(dst);
    const auto* s8 = reinterpret_cast<const unsigned char*>(src);
    // `dst` is shared memory on every path that reaches here, so its alignment is read from the
    // shared-window offset -- the quantity the hardware checks -- rather than from the generic
    // address, which agrees with it only while the window base is itself aligned.
    const unsigned ds      = smem_addr(dst);
    const std::uintptr_t s = reinterpret_cast<std::uintptr_t>(src);
    // The widest unit BOTH addresses satisfy. A pair sharing only 4-byte alignment still moves four
    // bytes at a time, and only a pair sharing nothing falls back to single bytes. Every unit moves
    // the same bytes, so this changes no numerical path -- it only stops a half-aligned pair from
    // costing sixteen scalar moves where four suffice.
    int unit = 1;
    if ((ds & 3U) == 0U && (s & 3U) == 0U) { unit = 4; }
    if ((ds & 7U) == 0U && (s & 7U) == 0U) { unit = 8; }
    if ((ds & 15U) == 0U && (s & 15U) == 0U) { unit = 16; }
    if (unit > Bytes) { unit = Bytes; }
    if (unit < Bytes) {
        NINFER_ALIGN_REPORT("copy_bytes", Bytes, ds & static_cast<unsigned>(Bytes - 1),
                            s, s & static_cast<std::uintptr_t>(Bytes - 1));
    }
    int i = 0;
    if (unit == 16) {
        for (; i + 16 <= Bytes; i += 16) {
            *reinterpret_cast<uint4*>(d8 + i) = *reinterpret_cast<const uint4*>(s8 + i);
        }
    } else if (unit == 8) {
        for (; i + 8 <= Bytes; i += 8) {
            *reinterpret_cast<uint2*>(d8 + i) = *reinterpret_cast<const uint2*>(s8 + i);
        }
    } else if (unit == 4) {
        for (; i + 4 <= Bytes; i += 4) {
            *reinterpret_cast<unsigned*>(d8 + i) = *reinterpret_cast<const unsigned*>(s8 + i);
        }
    }
#pragma unroll
    for (; i < Bytes; ++i) { d8[i] = s8[i]; }
}

template <int Bytes>
__device__ __forceinline__ void zero_bytes(void* dst) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "zero_bytes supports 4, 8, or 16 bytes");
    auto* d8 = reinterpret_cast<unsigned char*>(dst);
    const unsigned ds = smem_addr(dst);
    int unit = 1;
    if ((ds & 3U) == 0U) { unit = 4; }
    if ((ds & 7U) == 0U) { unit = 8; }
    if ((ds & 15U) == 0U) { unit = 16; }
    if (unit > Bytes) { unit = Bytes; }
    if (unit < Bytes) {
        NINFER_ALIGN_REPORT("zero_bytes", Bytes, ds & static_cast<unsigned>(Bytes - 1), ds,
                            ds & static_cast<unsigned>(Bytes - 1));
    }
    int i = 0;
    if (unit == 16) {
        for (; i + 16 <= Bytes; i += 16) {
            *reinterpret_cast<uint4*>(d8 + i) = make_uint4(0U, 0U, 0U, 0U);
        }
    } else if (unit == 8) {
        for (; i + 8 <= Bytes; i += 8) {
            *reinterpret_cast<uint2*>(d8 + i) = make_uint2(0U, 0U);
        }
    } else if (unit == 4) {
        for (; i + 4 <= Bytes; i += 4) {
            *reinterpret_cast<unsigned*>(d8 + i) = 0U;
        }
    }
#pragma unroll
    for (; i < Bytes; ++i) { d8[i] = 0; }
}
#endif // NINFER_SM75

// Turing's opt-in ceiling for per-block dynamic shared memory is 64 KiB, and the driver rejects a
// request of exactly 65536 bytes, so 64512 is the largest value cudaFuncSetAttribute accepts. The
// kernels in this fork are sized for the ~99 KiB Ampere budget, and an over-large request does not
// fail where it is made: cudaFuncSetAttribute returns cudaErrorInvalidValue and the failure
// surfaces later, as a launch error against whichever stream runs next. Every arena constant
// asserts the ceiling here, so the compiler names the offending kernel instead of the driver.
// The largest per-block dynamic shared-memory request a kernel in this tree may make.
//
// The unified Linear templates were sized against Ampere's and Ada's ~99 KiB opt-in budget. Turing
// admits 64 KiB per block and the driver rejects a request of exactly 65536 bytes, so 64512 is the
// largest value cudaFuncSetAttribute accepts there. Every arena constant in the tree asserts its
// ceiling against this value, so a kernel that would be mis-sized on the target build is named by the
// compiler instead of failing its launch at run time.
#if defined(NINFER_SM75)
inline constexpr int kMaxDynamicSharedMemoryBytes = 64512;
inline constexpr int kTuringDynamicSmemLimit       = kMaxDynamicSharedMemoryBytes;
#define NINFER_ASSERT_TURING_SMEM(expr)                                                            \
    static_assert((expr) <= ::ninfer::ops::kMaxDynamicSharedMemoryBytes,                            \
                  "dynamic shared memory exceeds Turing's 64512-byte per-block limit")
#else
inline constexpr int kMaxDynamicSharedMemoryBytes = 99 * 1024;
#define NINFER_ASSERT_TURING_SMEM(expr) static_assert(true, "")
#endif

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
#if defined(NINFER_SM75)
    // Turing has no cp.async. Stage synchronously at the same width instead, through the
    // alignment-safe helper: cp.async never required the caller's source to be aligned to the copy
    // width, so a width-matched typed vector move is only legitimate when the address actually is.
    // cp_commit/cp_wait below retire nothing because nothing is ever in flight, and every caller
    // already synchronises with __syncthreads() before reading the stage.
    copy_bytes<Bytes>(smem_dst, gmem_src);
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
#if defined(NINFER_SM75)
    // The asynchronous zfill copies min(src_bytes, Bytes) and zeroes the remainder; a fully
    // out-of-range source (src_bytes <= 0) zeroes the whole destination. A partial source holds as
    // many valid bytes as it holds, so it stays byte-wise; the whole-copy and whole-zero arms go
    // through the alignment-safe helpers, where a misaligned source is exactly what used to fault.
    if (src_bytes >= Bytes) {
        copy_bytes<Bytes>(smem_dst, gmem_src);
    } else if (src_bytes <= 0) {
        zero_bytes<Bytes>(smem_dst);
    } else {
        auto* dst       = reinterpret_cast<unsigned char*>(smem_dst);
        const auto* src = reinterpret_cast<const unsigned char*>(gmem_src);
#pragma unroll
        for (int i = 0; i < Bytes; ++i) {
            dst[i] = (i < src_bytes) ? src[i] : static_cast<unsigned char>(0);
        }
    }
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
#endif
}

__device__ __forceinline__ void cp_commit() {
#if !defined(NINFER_SM75)
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if !defined(NINFER_SM75)
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
#if defined(NINFER_SM75)
    cp_async<Bytes>(smem_dst, gmem_src);
#else
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
#endif
}

__device__ __forceinline__ void pipe_commit() {
#if !defined(NINFER_SM75)
    __pipeline_commit();
#endif
}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
#if !defined(NINFER_SM75)
    __pipeline_wait_prior(Groups);
#endif
}

} // namespace ninfer::ops
