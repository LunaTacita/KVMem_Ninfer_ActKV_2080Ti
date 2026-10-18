#pragma once

// CUDA 12.6 has <cuda_fp8.h> but no <cuda_fp4.h>: the FP4 types and the cvt.rn.satfinite.e2m1x2
// PTX conversion only arrived with CUDA 12.8 / sm_90. The nvfp4 codecs are still part of this
// build (sparse_moe and the causal-cache nvfp4/k8v4 kernels all reach them), so this header
// supplies the missing type rather than dropping the capability.
//
// On a toolchain that does ship <cuda_fp4.h> the vendor header is used unchanged and nothing here
// is defined, which keeps the official implementation the single source of truth wherever it
// exists.
//
// LOCAL ADAPTATION (toolchain, not part of any architecture port), 2026-10-12.
// NOTE: the values below are not yet checked against tests/artifact/test_nvfp4_numeric.py. The
// current delivery target (Ternary-Bonsai PQ2_0_G128) never enters an NVFP4 path, so this file
// exists to keep the translation units compiling on CUDA 12.6; the numerics should be verified
// against that test before any nvfp4 artifact is used in anger.

#if defined(__has_include)
#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#define NINFER_HAS_CUDA_FP4 1
#endif
#endif

#ifndef NINFER_HAS_CUDA_FP4

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// The observable contract the call sites rely on: a public `__x` byte holding two E2M1 nibbles
// with the low nibble carrying the `.x` element, plus conversion to and from float2.
//
// E2M1 is one sign bit over three magnitude bits, and its eight magnitudes are
// {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Encoding is round-to-nearest with ties resolved to the even code,
// and saturating outside the representable range -- the behaviour of cvt.rn.satfinite.
struct __nv_fp4x2_e2m1 {
    unsigned char __x;

    __device__ __forceinline__ __nv_fp4x2_e2m1() : __x(0) {}
    __device__ __forceinline__ __nv_fp4x2_e2m1(const __nv_fp4x2_e2m1&) = default;
    __device__ __forceinline__ __nv_fp4x2_e2m1& operator=(const __nv_fp4x2_e2m1&) = default;

    __device__ __forceinline__ explicit __nv_fp4x2_e2m1(float2 value) : __x(pack(value)) {}

    // float -> one E2M1 nibble (magnitude in bits 0..2, sign in bit 3).
    __device__ __forceinline__ static unsigned char encode(float value) {
        const unsigned sign = static_cast<unsigned>(__float_as_uint(value) >> 31) << 3;
        float magnitude     = fabsf(value);
        // Saturate past the largest magnitude. NaN takes this branch as well, which is what
        // satfinite does; E2M1 has no NaN encoding.
        if (!(magnitude <= 6.0f)) { magnitude = 6.0f; }

        unsigned code;
        if (magnitude <= 0.25f) {
            code = 0; // 0.25 is the tie between 0.0 and 0.5; the even code is 0
        } else if (magnitude < 0.75f) {
            code = 1;
        } else if (magnitude <= 1.25f) {
            code = 2; // 0.75 and 1.25 tie between codes 1/2 and 2/3; the even code is 2
        } else if (magnitude < 1.75f) {
            code = 3;
        } else if (magnitude <= 2.5f) {
            code = 4; // 1.75 and 2.5 tie; the even code is 4
        } else if (magnitude < 3.5f) {
            code = 5;
        } else if (magnitude <= 5.0f) {
            code = 6; // 3.5 and 5.0 tie; the even code is 6
        } else {
            code = 7;
        }
        return static_cast<unsigned char>(sign | code);
    }

    // One E2M1 nibble -> float. The sign is applied through the bit pattern so that the negative
    // zero code (0b1000) decodes to -0.0f rather than being folded to +0.0f.
    __device__ __forceinline__ static float decode(unsigned nibble) {
        float magnitude;
        switch (nibble & 7u) {
        case 0: magnitude = 0.0f; break;
        case 1: magnitude = 0.5f; break;
        case 2: magnitude = 1.0f; break;
        case 3: magnitude = 1.5f; break;
        case 4: magnitude = 2.0f; break;
        case 5: magnitude = 3.0f; break;
        case 6: magnitude = 4.0f; break;
        default: magnitude = 6.0f; break;
        }
        return __uint_as_float(__float_as_uint(magnitude) | ((nibble & 8u) << 28));
    }

    // float2 -> one packed E2M1x2 byte.
    __device__ __forceinline__ static unsigned char pack(float2 value) {
        return static_cast<unsigned char>(encode(value.x) |
                                          (static_cast<unsigned>(encode(value.y)) << 4));
    }

    __device__ __forceinline__ operator float2() const {
        return make_float2(decode(static_cast<unsigned>(__x & 0x0Fu)),
                           decode(static_cast<unsigned>((__x >> 4) & 0x0Fu)));
    }

    // The KV cache scale path dequantizes with half2 arithmetic and converts the packed code
    // straight to __half2 (kv_cache/nvfp4_group16_codec.cuh), so the portable type has to expose
    // the same conversion the vendor type does. Both decoded values are exactly representable in
    // FP16, so this widening is lossless.
    __device__ __forceinline__ operator __half2() const {
        const float2 value = static_cast<float2>(*this);
        return __floats2half2_rn(value.x, value.y);
    }
};

// Software replacement for the `mov.b32 {b0,b1,b2,b3}` byte packing the PTX form performed: index
// 0 lands in the low byte.
__device__ __forceinline__ std::uint32_t nvfp4_pack_e2m1x8_portable(const float2 (&values)[4]) {
    std::uint32_t codes = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        codes |= static_cast<std::uint32_t>(__nv_fp4x2_e2m1::pack(values[i]))
                 << (8 * static_cast<unsigned>(i));
    }
    return codes;
}

#endif // !NINFER_HAS_CUDA_FP4
