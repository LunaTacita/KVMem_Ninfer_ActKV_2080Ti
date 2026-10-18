#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = prepare_wy_wu;

template <int KPanelCols, int WuPanelCols, int BlockWarps>
cudaError_t launch_fixed(const prepare_wy_wu_config& cfg, dim3 grid, head_map qk_map) {
    using dims               = kernel::kernel_dims<KPanelCols, WuPanelCols>;
    constexpr int smem_bytes = dims::SMEM_FLOATS * static_cast<int>(sizeof(float));
    NINFER_ASSERT_TURING_SMEM(smem_bytes);
    constexpr int threads    = BlockWarps * ninfer::ops::kWarpSize;

    cudaError_t err =
        cudaFuncSetAttribute(kernel::prepare_wy_wu_kernel<KPanelCols, WuPanelCols, BlockWarps>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }

    kernel::prepare_wy_wu_kernel<KPanelCols, WuPanelCols, BlockWarps>
        <<<grid, dim3(threads, 1, 1), smem_bytes, cfg.stream>>>(cfg.k, cfg.k_inv_norm, cfg.v,
                                                                cfg.g_in, cfg.beta, cfg.W, cfg.U,
                                                                cfg.g_cumsum_out, qk_map);
    return cudaGetLastError();
}

// Diagnosis, silent unless something is wrong. The kernel stages K and V with 16-byte copies and
// reads W/U panels with 8-byte vector loads, so every base handed to it must be 16-byte aligned.
// The first revision of this probe checked only k/v/W/U; g_in, beta and g_cumsum_out are operands
// too, and a misaligned one faults exactly the same way. The shared probe in launch.h names every
// offending operand and prints nothing when all of them are aligned.
void probe_base_pointers(const prepare_wy_wu_config& cfg) {
    report_misalignment("prepare_wy_wu",
                        {{"k", cfg.k},
                         {"v", cfg.v},
                         {"g_in", cfg.g_in},
                         {"beta", cfg.beta},
                         {"W", cfg.W},
                         {"U", cfg.U},
                         {"g_cumsum_out", cfg.g_cumsum_out}},
                        cfg.H_qk, cfg.H_v, cfg.L);
}

} // namespace

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg) {
    stage_validator v{"launch_prepare_wy_wu", cfg.H_qk, cfg.H_v, cfg.L};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.k == nullptr || cfg.v == nullptr || cfg.g_in == nullptr || cfg.beta == nullptr ||
        cfg.W == nullptr || cfg.U == nullptr || cfg.g_cumsum_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    probe_base_pointers(cfg);

    const auto qk_map     = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const std::int64_t NT = cfg.L / BT;
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(NT, cfg.H_v));

    const dim3 grid(static_cast<unsigned>(NT), static_cast<unsigned>(cfg.H_v), 1);
    if (cfg.H_v == 32) { return launch_fixed<32, 16, 4>(cfg, grid, qk_map); }
    return launch_fixed<64, 32, 8>(cfg, grid, qk_map);
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
