#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

// ldmatrix loads one 16-byte row per lane, so every lane's shared-memory address must be 16-byte
// aligned. That requirement is the instruction's on every architecture, but this port newly puts it
// at risk on Turing: the shared-memory arenas were resized to fit the 64 KiB per-block ceiling, and
// a row stride that moved with them can leave a row start off the boundary. Check rather than
// assume; the check compiles away entirely unless NINFER_ALIGN_DIAG is on, and when it is on it
// reports the shared-window remainder so the offending row base is identifiable from the log.
#if NINFER_ALIGN_DIAG
#define NINFER_LDMATRIX_ALIGN_CHECK(what, addr)                                                    \
    do {                                                                                           \
        if (((addr) & 15U) != 0U) {                                                                \
            NINFER_ALIGN_REPORT((what), 16U, (addr) & 15U, (addr), (addr) & 15U);                  \
        }                                                                                          \
    } while (0)
#else
#define NINFER_LDMATRIX_ALIGN_CHECK(what, addr) ((void)0)
#endif

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    NINFER_LDMATRIX_ALIGN_CHECK("ldmatrix_x2", addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    NINFER_LDMATRIX_ALIGN_CHECK("ldmatrix_x4", addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    NINFER_LDMATRIX_ALIGN_CHECK("ldmatrix_x2_t", addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    NINFER_LDMATRIX_ALIGN_CHECK("ldmatrix_x4_t", addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// ---------------------------------------------------------------------------------------------
// Turing (sm_75) downgrades.
//
// sm_75 does have tensor cores, but only the Turing shapes: f16 m16n8k8 and s8 m8n8k16. It has no
// m16n8k16 / m16n8k32 at all, no bf16 tensor core (f16 only), no tf32, and none of the fp8 / nvfp4
// `kind::` qualifiers. Everything below keeps this file's public signatures and accumulator
// register order unchanged, so not one caller has to move.
//
// Both shapes hand their A registers over in the same order: a0=(M0,K0) a1=(M1,K0) a2=(M0,K1)
// a3=(M1,K1), where M0 is the first eight rows, M1 the second eight, K0 the first half of the
// contraction and K1 the second. An earlier revision of this file recorded an s8-specific
// "k-major" order (a1=(M0,K1) a2=(M1,K0)) and paired the four m8n8k16 steps against it, which
// cross-paired the two destination halves and quietly corrupted every mma_s8 caller in the tree.
// The order is taken from the callers -- t2_small_t_i8.cuh names the registers top/bottom x
// even/odd, rowsplit_a8_mma.cuh loads r0/r0+8 x byte/byte+4, and t2_prefill_i8.cuh passes
// d0/d1 x 2s/2s+1 -- and those are the authority here, not the f16 shape's letters.
// ---------------------------------------------------------------------------------------------

#if defined(NINFER_SM75)
// Convert a register holding two bf16 values into the same two values in f16. bf16 has 8 exponent
// and 7 mantissa bits, f16 has 5 and 10, so the narrowing is exact while the magnitude stays
// inside f16's range (|x| <= 65504); outside it the value saturates exactly as any f16 conversion.
__device__ __forceinline__ unsigned bf162_to_f162(unsigned b) {
    const auto* bf_ptr = reinterpret_cast<const __nv_bfloat162*>(&b);
    const float2 f     = __bfloat1622float2(*bf_ptr);
    const half2 h      = __float22half2_rn(f);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Turing-native m16n8k8: the two A registers are M0 and M1 over the single 8-wide K slice.
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_f16_f16acc_m16n8k8(unsigned& d0, unsigned& d1, unsigned a0,
                                                       unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}
#endif

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM75)
    // No bf16 tensor core on Turing: narrow both operands to f16 and use the f16 tensor core.
    const unsigned e0 = bf162_to_f162(a0);
    const unsigned e1 = bf162_to_f162(a1);
    const unsigned e2 = bf162_to_f162(a2);
    const unsigned e3 = bf162_to_f162(a3);
    const unsigned g0 = bf162_to_f162(b0);
    const unsigned g1 = bf162_to_f162(b1);
    mma_f16_m16n8k8(c0, c1, c2, c3, e0, e1, g0);
    mma_f16_m16n8k8(c0, c1, c2, c3, e2, e3, g1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(NINFER_SM75)
    // m16n8k16 -> two Turing m16n8k8: the low K half takes (a0,a1) with b0 and the high half
    // (a2,a3) with b1, matching the .f16 fragment order recorded above.
    mma_f16_m16n8k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16 accumulation: twice the FP32-accumulator rate on GeForce parts. d0 holds the row-gid pair
// and d1 the row-gid+8 pair of the fragment, in the same order as mma_f16's c0..c3.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                               unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
#if defined(NINFER_SM75)
    mma_f16_f16acc_m16n8k8(d0, d1, a0, a1, b0);
    mma_f16_f16acc_m16n8k8(d0, d1, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // m16n8k32 -> four Turing m8n8k16, in the A fragment order recorded above: a0/a2 carry M0
    // against K0/K1, a1/a3 carry M1, and c0/c1 accumulate the M0 half of the destination while
    // c2/c3 accumulate M1. Each K slice therefore stays on its own destination half.
    mma_s8_m8n8k16(c0, c1, a0, b0);
    mma_s8_m8n8k16(c0, c1, a2, b1);
    mma_s8_m8n8k16(c2, c3, a1, b0);
    mma_s8_m8n8k16(c2, c3, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // Upstream's block-scaled form (RTX 5090: the plain form accumulates FP32 at half rate).
    // UE8M0 0x7f is exactly 1, so this preserves the raw E4M3 dot product and keeps the callers'
    // existing row scales in FP32 after the MMA, with no operand repacking.
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // Turing has no tf32 tensor core. Reproduce the m16n8k8 row.col fp32 tile exactly with warp
    // shuffles and SIMT FMAs. Thread T holds r = T/4 in [0,8) and t = T%4 in [0,4), so lane 4r+k
    // carries A(r,k) in fa0, A(r+8,k) in fa1, A(r,k+4) in fa2 and A(r+8,k+4) in fa3, while lane
    // 8t+k carries B(k,2t) in fb0 and B(k+4,2t) in fb1. The target C registers are c0=C(r,2t),
    // c1=C(r,2t+1), c2=C(r+8,2t), c3=C(r+8,2t+1).
    const int lane    = threadIdx.x & 31;
    const int r       = lane >> 2;
    const int t       = lane & 3;
    const int base_a  = 4 * r;
    const int base_b0 = 8 * t;
    const int base_b1 = 8 * t + 4;

    const float fa0 = __uint_as_float(a0);
    const float fa1 = __uint_as_float(a1);
    const float fa2 = __uint_as_float(a2);
    const float fa3 = __uint_as_float(a3);
    const float fb0 = __uint_as_float(b0);
    const float fb1 = __uint_as_float(b1);

    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;

#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
        const float a_r_k   = __shfl_sync(0xffffffffU, fa0, base_a + k_idx);
        const float a_r8_k  = __shfl_sync(0xffffffffU, fa1, base_a + k_idx);
        const float a_r_k4  = __shfl_sync(0xffffffffU, fa2, base_a + k_idx);
        const float a_r8_k4 = __shfl_sync(0xffffffffU, fa3, base_a + k_idx);

        const float b_n0_k  = __shfl_sync(0xffffffffU, fb0, base_b0 + k_idx);
        const float b_n0_k4 = __shfl_sync(0xffffffffU, fb1, base_b0 + k_idx);
        const float b_n1_k  = __shfl_sync(0xffffffffU, fb0, base_b1 + k_idx);
        const float b_n1_k4 = __shfl_sync(0xffffffffU, fb1, base_b1 + k_idx);

        sum0 += a_r_k * b_n0_k + a_r_k4 * b_n0_k4;
        sum1 += a_r_k * b_n1_k + a_r_k4 * b_n1_k4;
        sum2 += a_r8_k * b_n0_k + a_r8_k4 * b_n0_k4;
        sum3 += a_r8_k * b_n1_k + a_r8_k4 * b_n1_k4;
    }

    c0 += sum0;
    c1 += sum1;
    c2 += sum2;
    c3 += sum3;
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
#if defined(NINFER_SM75)
    // `kind::mxf4nvf4` is sm_120a-only and Turing has no route that reproduces it. Trap for the
    // same reason as mma_fp8_e4m3 above: a no-op would quietly corrupt every score it feeds.
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1; (void)sfa; (void)sfb;
    asm volatile("trap;");
#else
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#endif
}

} // namespace ninfer::ops
