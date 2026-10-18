#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N2048K16384;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
#if defined(NINFER_SM75)
// Turing's 64 KiB per-block opt-in ceiling cannot hold KWarps=16 at these token tiles:
// kSharedBytes = Stages * (1088*KWarps + 128*KWarps*BlockTokens) against a 64512-byte limit, so
// KWarps=16 is over budget from BlockTokens=24 up and even the two-stage BlockTokens=8 tile needs
// 67584. Halving KWarps keeps BlockTokens spanning the whole token band -- no launch reads past its
// tile -- and splits the same K reduction in another order. StaticK stays a multiple of KWarps*64:
// 16384 % (8*64) == 0.
using C8 = Q8A16SlicedKMmaSchedule<8, 8, 2, 1, Access::Shared>;
#else
using C8 = Q8A16SlicedKMmaSchedule<8, 16, 2, 1, Access::Shared>;
#endif
using Wide32   = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Access::Shared>;
using Medium16 = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Access::Shared>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
#if defined(NINFER_SM75)
using C24 =
    Q8A16SlicedKMmaSchedule<24, 8, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C32 =
    Q8A16SlicedKMmaSchedule<32, 8, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8A16SlicedKMmaSchedule<40, 8, 1, 1, Access::Shared, Cache::cg, Cache::cg,
                                    Stage::RuntimeActive>;
#else
using C24 =
    Q8A16SlicedKMmaSchedule<24, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C32 =
    Q8A16SlicedKMmaSchedule<32, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8A16SlicedKMmaSchedule<40, 16, 1, 1, Access::Shared, Cache::cg, Cache::cg,
                                    Stage::RuntimeActive>;
#endif
using C48 =
    Q8A16SlicedKMmaSchedule<48, 8, 1, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;

void launch_grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
#if defined(NINFER_SM75)
    // kSharedBytes = Stages * (16*KWarps*64 + KWarps*BlockTokens*128) = 69632 at two stages, over
    // Turing's 64512-byte per-block ceiling; one stage needs 34816. KWarps is already at its minimum
    // of 2 and BlockTokens must stay 128 to cover the token band.
    using Schedule =
        Q8A16GroupedSlicedKMmaSchedule<128, 2, 4, 1, 1, 0, Cache::cg, Cache::cg, true, false>;
#else
    using Schedule =
        Q8A16GroupedSlicedKMmaSchedule<128, 2, 4, 2, 1, 0, Cache::cg, Cache::cg, true, false>;
#endif
    launch_q8_a16_grouped_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n2048_k16384_unified(std::int32_t tokens) {
    if (tokens == 1) return launch_q8_a16_gemv_r4_w1_k16384;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 64) return launch_q8_a16_sliced<Geometry, 32, Wide32>;
    if (tokens <= 80) return launch_q8_a16_sliced<Geometry, 16, Medium16>;
    if (tokens <= 128) return launch_grouped;

    // Broad throughput regions; each selected MMA handles its own complete and partial tiles.
    if (tokens <= 384) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 480) return launch_q8_a16_mma_r32_t96;
    if (tokens <= 640) return launch_q8_a16_mma_r32_t128;
    if (tokens <= 704) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 960) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 1344) return launch_q8_a16_mma_r128_t64;
    if (tokens <= 1680) return launch_q8_a16_mma_r128_t80;
    if (tokens <= 2016) return launch_q8_a16_mma_r64_t96;
    if (tokens <= 2112) return launch_q8_a16_mma_r96_t96;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
