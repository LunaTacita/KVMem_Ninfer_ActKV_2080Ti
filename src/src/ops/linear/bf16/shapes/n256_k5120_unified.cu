#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes_unified.h"
#include "ops/linear/bf16/bf16_launch_unified.cuh"

namespace ninfer::ops::detail::unified {
Bf16Launch select_bf16_n256_k5120(std::int32_t tokens) {
#if defined(NINFER_SM75)
    // Turing has neither TMA nor a 64 KiB+ per-block dynamic shared memory budget. Out are the
    // sliced-K tiles at KWarps 16 and 8 (98304 and 65536 bytes), the K=256 MMA tile at three stages
    // (98304) and every TMA schedule. What fits is the sliced tiles with KWarps halved (49152 and
    // 32768) and the K=128 MMA tile (49152); 5120 stays a multiple of each KWarps*64 used here.
    if (tokens <= 76)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8>, 5120>>;
    if (tokens <= 160)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4>, 5120>>;
    if (tokens <= 640) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
    return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
#else
    if (tokens <= 76)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 16>, 5120>,
            pdl::Dependency::Programmatic>;
    if (tokens <= 160)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8>, 5120>,
            pdl::Dependency::Programmatic>;
    if (tokens <= 640) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 5120>>;
    if (tokens <= 1024) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 5120>>;
    if (tokens <= 1280)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T32K64S3, 5120>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 5120>>;
#endif
}
} // namespace ninfer::ops::detail::unified
