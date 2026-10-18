#pragma once

// One source of truth for "how wide may a single-CTA small-T tile be on this architecture".
//
// CUDA requires `static __shared__ + dynamic shared memory <= cudaDevAttrMaxSharedMemoryPerBlockOptin`
// (64 KiB on Turing, of which the driver rejects exactly 65536, so 64512 is the largest accepted
// request). `NINFER_ASSERT_TURING_SMEM` only bounds the DYNAMIC half, so a kernel whose arena is
// within budget can still be unlaunchable once its own q_s/p_s/alpha_s arenas are added: the failure
// then surfaces as cudaErrorInvalidValue from cudaFuncSetAttribute, which is a process abort rather
// than a catchable error.
//
// Measured with `cuobjdump --dump-resource-usage` on this build (static SHARED:), against the
// launchers' dynamic requests in small_t_{fp8,k8v4,nvfp4}.cu:
//
//   family   RowTiles=1            RowTiles=2            RowTiles=3            dynamic
//   fp8      10688 ->  59840 ok    20992 -> 70144 OVER   31296 -> 80448 OVER   6*Bc*D
//   k8v4     11136 ->  56192 ok    21440 -> 66496 OVER   31744 -> 76800 OVER   11*Bc*D/2
//   nvfp4    11584 ->  36160 ok    21888 -> 46464 ok     32192 -> 73152 OVER   (3 or 5)*Bc*D
//
// RowTiles is ceil(TokenTile * Geometry::GroupSize / 16), i.e. it is a consequence of the query-row
// tile width, not an independent knob; it cannot be "shrunk" for a given width without changing what
// the kernel computes. The dynamic arena is a fixed multi-buffer layout (see the comments in the
// launchers), so neither half can be reduced. The launchable configuration left on Turing is
// therefore RowTiles == 1, and the affected widths are routed to the prompt kernel, which is a
// complete implementation of the same attention (dequantize, MMA, online softmax) with no static
// arena of its own.
//
// This header deliberately matches the numbers the routing layer needs to make the same decision
// without duplicating them: change a limit here and both the launcher template instantiation and
// causal_attention_resolve_route follow.

#include <cstdint>

namespace ninfer::ops::detail {

// The largest query-row tile a single small-T CTA may cover on the architecture being compiled for.
//
// Every `small_t_*_tiled_kernel` instantiation that would exceed the per-block shared-memory ceiling
// is excluded from the sm75 build entirely, so it can neither be launched nor abort the process at
// cudaFuncSetAttribute time. `Geometry::GroupSize` is 6 on the 24-head geometry and 8 on the
// 16-head one (see AttentionHeadMapping), so a width limit below maps to RowTiles == 1 on both.
#if defined(NINFER_SM75)
// FP8 keeps a real dequantized BF16 copy of K beside the raw FP8 stage, so its fixed 6*Bc*D = 49152
// byte arena leaves room for RowTiles == 1 only.
inline constexpr std::int32_t kSmallTFp8MaximumWidth = 2;
// K8V4 stages k_fp8 + k_b16 + v_nvfp4 + v_f16 = 5.5*Bc*D = 45056 bytes, same conclusion.
inline constexpr std::int32_t kSmallTK8V4MaximumWidth = 2;
// NVFP4's arena is the smallest (3*Bc*D = 24576 bytes while V reuses K's f16 stage), so the
// two-row tile (21888 + 24576 = 46464) still fits and only RowTiles == 3 must go.
inline constexpr std::int32_t kSmallTNvfp4MaximumWidth = 4;
#else
// Ampere and Ada admit ~99 KiB per block; every tile this fork instantiates fits there.
inline constexpr std::int32_t kSmallTFp8MaximumWidth   = 8;
inline constexpr std::int32_t kSmallTK8V4MaximumWidth  = 8;
inline constexpr std::int32_t kSmallTNvfp4MaximumWidth = 8;
#endif

} // namespace ninfer::ops::detail
