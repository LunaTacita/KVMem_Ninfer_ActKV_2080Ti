#include "ops/linear/fp8/fp8_shapes_unified.h"
#include "ops/linear/fp8/fp8_launch_unified.cuh"
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {
using Geometry = Fp8Geometry<16384, 5120>;
using Gemv     = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
// Seven resident CTAs keep the 1,024 row tiles in one wave without register spills.
using Sliced16 =
    Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, 1>;
#if !defined(NINFER_SM75)
// The A8 TMA schedules need more than Turing's 64512 B per-block shared memory and emit sm_80+
// MMA, and A8 is never selected on this target anyway (allows_a8_int forces
// LinearPolicy::A16Only, see ops/linear/common/route_table.h). The aliases are guarded as well
// because instantiating the schedule by itself trips fp8_schedule.cuh's shared-memory assert.
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
#endif

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Sliced16>(x, weight, out, stream);
#if defined(NINFER_SM75)
    // Fp8SlicedInstance<32,8,2> needs 82944 bytes at two activation stages, over Turing's
    // 64512-byte per-block ceiling; the KWarps=4 single-stage tile below needs 20736 and fits.
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                             stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                             stream);
#else
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
#endif
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    if (tokens <= 96)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
#if defined(NINFER_SM75)
    // A8 is never selected on Turing - allows_a8_int forces LinearPolicy::A16Only - and its TMA
    // schedules cannot be built for this target, so keep the Op boundary link-complete and reject
    // the route precisely, exactly as fp8_sm86_stubs.cpp does for the other A8 entry points.
    (void)x; (void)weight; (void)out; (void)scratch; (void)stream;
    throw std::runtime_error("FP8 A8 execution requires an sm_100a or sm_120a GPU");
#else
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R32K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 64)
#if defined(NINFER_SM75)
        // Fp8A8T64R128K256 stages kStages*(kBlockTokens+kBlockRows)*kBlockK bytes, over Turing's
        // 64512-byte per-block ceiling; the K=128 tile stages half that and is already instantiated
        // for the widest band below.
        return launch_fp8_a8<Geometry, Fp8A8T64R128K128>(x, weight, out, scratch, stream);
#else
        return launch_fp8_a8<Geometry, Fp8A8T64R128K256>(x, weight, out, scratch, stream);
#endif
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma192x128>(x, weight, out, scratch, stream);
    // Smaller output tiles leave only two full-K tiles to split near the 512-token anchor.
    if (x.ne[1] > 384 && x.ne[1] <= 512)
        return launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
#endif
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) {
#if defined(NINFER_SM75)
    // Matches allows_a8_int's forced LinearPolicy::A16Only: no A8 route exists on this target.
    (void)max_tokens;
    return false;
#else
    return max_tokens >= 17;
#endif
}

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
#if defined(NINFER_SM75)
    (void)max_tokens;
    return 0; // No A8 launch, so no TMA split-K partials to reserve.
#else
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
#endif
}
} // namespace

const Fp8LinearShape kFp8N16384K5120{16384,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail::unified
