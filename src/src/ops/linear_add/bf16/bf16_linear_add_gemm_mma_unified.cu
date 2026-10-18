#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/linear_add/bf16/bf16_linear_add_plan_unified.h"

namespace ninfer::ops::detail::unified {
namespace {
template <class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    launch_bf16_a16_mma<Bf16ScheduleInstance<Schedule, 6144>>(
        bf16_a16_operands(x, weight), LinearBf16Output{data, weight.n},
        LinearResidualAddEpilogue{{data, weight.n}}, stream);
}
} // namespace

void bf16_linear_add_aggregate_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                          cudaStream_t stream) {
#if defined(NINFER_SM75)
    // Bf16A16MmaR32T32K256S3 needs 98304 bytes of dynamic shared memory, over Turing's 64512-byte
    // per-block ceiling; Bf16A16MmaR32T32K128S3 needs 49152 and fits. See n256_k5120_unified.cu.
    using UpTo32 = Bf16A16MmaR32T32K128S3;
#else
    using UpTo32 = Bf16A16MmaR32T32K256S3;
#endif
    using UpTo48 = Bf16A16MmaR32T32K192S2;

    if (x.ne[1] <= 32)
        launch<UpTo32>(x, weight, residual, stream);
    else
        launch<UpTo48>(x, weight, residual, stream);
}

void bf16_linear_add_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                cudaStream_t stream) {
    auto* data     = static_cast<__nv_bfloat16*>(residual.data);
    const auto p   = bf16_a16_operands(x, weight);
#if defined(NINFER_SM75)
    // Turing has no TMA and no 64 KiB+ per-block dynamic shared memory: the TMA kernels are not
    // compiled for sm_75, and Bf16A16TmaR64T64K128S2 would stage 65536 bytes anyway. The MMA tile
    // stages 49152 at K=128 and fits.
    launch_bf16_a16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 6144>>(
        p, LinearBf16Output{data, weight.n}, LinearResidualAddEpilogue{{data, weight.n}}, stream);
#else
    const auto tma = [&]<class S>() {
        launch_bf16_a16_tma_mma<Bf16ScheduleInstance<S, 6144>>(
            p, LinearBf16Output{data, weight.n}, LinearResidualAddEpilogue{{data, weight.n}},
            stream);
    };
    if (p.tokens <= 96)
        tma.template operator()<Bf16A16TmaR64T32K64S3>();
    else if (p.tokens <= 128)
        tma.template operator()<Bf16A16TmaR64T64K128S2>();
    else if (p.tokens <= 256)
        tma.template operator()<Bf16A16TmaR64T64K64S3>();
    else
        tma.template operator()<Bf16A16TmaR64T128K64S2>();
#endif
}
} // namespace ninfer::ops::detail::unified
