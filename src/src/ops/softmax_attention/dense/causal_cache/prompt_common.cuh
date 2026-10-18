#pragma once

// Shared causal-attention dimensions and leaf PTX helpers used by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

// __maxnreg__ is only honoured on compute capability 8.0 and above, and nvcc rejects it outright
// for a Turing target. Define it away there: the prompt kernels then let ptxas choose the register
// count instead of pinning it, which costs occupancy tuning on sm_75 and changes nothing on the
// architectures that do support the qualifier.
#if defined(NINFER_SM75)
#define NINFER_MAXNREG(registers)
// nvcc rejects __maxnreg__ on a Turing target, and without it ptxas picks more registers than a
// 512-thread prompt block can fit in the SM's 64K-register file: measured 182 registers x 512
// threads = 93184 > 65536, which the driver reports at launch as cudaErrorLaunchOutOfResources.
// __launch_bounds__(Threads, 1) is accepted on Turing and imposes the same ceiling implicitly:
// at most 65536/Threads registers per thread.
#define NINFER_LAUNCH_BOUNDS_FOR_MAXNREG(threads) __launch_bounds__(threads, 1)
#else
#define NINFER_MAXNREG(registers) __maxnreg__(registers)
#define NINFER_LAUNCH_BOUNDS_FOR_MAXNREG(threads)
#endif

namespace ninfer::ops {

inline constexpr int kCausalPromptHeadDim = 256;

inline constexpr int kCausalPromptBr        = 64;
#if defined(NINFER_SM75)
// Turing's per-block dynamic shared-memory ceiling is 64512 bytes, less than half the ~99 KiB this
// fork's prompt tiles are sized for. Only the key tile shrinks: Br fixes RowTiles and with it the
// consumer split (DConsumers == Warps / (Br/16)), so moving Br would break that contract.
inline constexpr int kCausalPromptBc        = 16;
#else
inline constexpr int kCausalPromptBc        = 64;
#endif
inline constexpr int kCausalPromptThreads   = 128;
inline constexpr int kCausalPromptSmemBytes = (kCausalPromptBr + 2 * kCausalPromptBc) *
                                              kCausalPromptHeadDim *
                                              static_cast<int>(sizeof(__nv_bfloat16));
NINFER_ASSERT_TURING_SMEM(kCausalPromptSmemBytes);

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_prompt_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kCausalPromptHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ void causal_prompt_zero_output_rows(__nv_bfloat16* out, int q_head,
                                                               int row_begin, int row_end, int tid,
                                                               int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * kCausalPromptHeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / kCausalPromptHeadDim;
        const int d   = element - (row - row_begin) * kCausalPromptHeadDim;
        out[causal_prompt_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int causal_prompt_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_prompt_store_byte_swizzled(Byte* tile, int row, int d,
                                                                  Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off = (row * (kCausalPromptHeadDim / 2) + causal_prompt_swz(row, col_b16)) * 2 + byte;
    tile[off]     = code;
}

template <int Columns>
__device__ __forceinline__ int causal_prompt_p_swz(int row, int col) {
    // The XOR has to be as wide as the log2 of the number of 8-element chunks in a row: 64 columns
    // -> 8 chunks -> row&7 (the generic branch below), 32 -> 4 chunks -> row&3, and 16 -> 2 chunks
    // -> row&1. The 16-column case MUST be spelled out: falling through to the generic branch XORs
    // with row&7, whose result reaches 7, so the target chunk can be up to 7 while a 16-column row
    // has only 2. Different (row, col) pairs then land on the same half -- and for the last rows
    // the computed address runs past p_s into the k_scale_s that immediately follows it, which the
    // next issue_kv_tile overwrites. sm_75 is the first build to use a 16-column P tile
    // (kCausalPromptBc and kCausalPromptI8Bc are both 16 there), which is why this is only now
    // reachable.
    if constexpr (Columns == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    if constexpr (Columns == 16) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
    return causal_prompt_swz(row, col);
}

__device__ __forceinline__ unsigned causal_prompt_swz_addr(unsigned lane_base, unsigned ck,
                                                           unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

} // namespace ninfer::ops
