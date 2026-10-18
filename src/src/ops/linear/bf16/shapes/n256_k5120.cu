#include "ops/linear/bf16/bf16_shapes.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/bf16/bf16_n256_k5120.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
void launch_chunk(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
                  std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = 256 / Schedule::kOutputRowsPerCta;
    const dim3 grid(kRowTiles, (tokens + Schedule::kTileTokens - 1) / Schedule::kTileTokens);
    configure_cuda_device_once([] {
        return cudaFuncSetAttribute(bf16_n256_k5120_mma_kernel<Schedule>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Schedule::kSharedBytes);
    });
    CUDA_CHECK(
        pdl::launch_consumer({dim3(grid), dim3(Schedule::kThreads), Schedule::kSharedBytes, stream},
                             bf16_n256_k5120_mma_kernel<Schedule>, x, weight, out, tokens));
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_grid(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kTileTokens, [&](int offset, int count) {
        launch_chunk<Schedule>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(offset) * 5120,
            static_cast<const __nv_bfloat16*>(weight.qdata),
            static_cast<__nv_bfloat16*>(out.data) + static_cast<std::int64_t>(offset) * 256, count,
            stream);
    });
}
} // namespace

Bf16Launch select_bf16_n256_k5120(std::int32_t tokens) {
    // Turing's 64 KiB per-block opt-in ceiling: kStagingBytes = 2*(16+kTileTokens)*KWarps*64*2, so
    // KWarps=16 stages 98304 and KWarps=8 stages 65536, both over 64512. KWarps=4 stages 24576 and
    // 32768, and 5120 % (4*64) == 0 keeps the K tiling exact.
#if defined(NINFER_SM75)
    if (tokens <= 76) return launch_grid<Bf16N256K5120MmaSchedule<4, 8>>;
    return launch_grid<Bf16N256K5120MmaSchedule<4, 16>>;
#else
    if (tokens <= 76) return launch_grid<Bf16N256K5120MmaSchedule<16, 8>>;
    return launch_grid<Bf16N256K5120MmaSchedule<8, 16>>;
#endif
}

} // namespace ninfer::ops::detail
