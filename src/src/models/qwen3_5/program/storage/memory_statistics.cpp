#include "models/qwen3_5/program/program_impl.h"
#include <cmath>
#include <cstdio>

namespace ninfer::models::qwen3_5::detail {
namespace {
void read_statistics(DeviceContext& device, const execution::MemoryStatistics& statistics,
                     void* destination, bool include_query) {
    auto* host = static_cast<std::byte*>(destination);
    const auto key_stride = std::size_t(statistics.key_width) * statistics.columns * sizeof(float);
    const auto query_stride = std::size_t(statistics.query_width) * sizeof(float);
    for (std::size_t rank = 0; rank < statistics.ranks.size(); ++rank) {
        const auto& shard = statistics.ranks[rank];
        if (shard.key_sums.data == nullptr) { continue; }
        DeviceBinding binding(device.rank(rank).device);
        const auto stream = device.stream_for_rank(rank);
        CUDA_CHECK(cudaMemcpyAsync(host + shard.first_layer * key_stride,
            shard.key_sums.data, shard.key_sums.bytes(), cudaMemcpyDeviceToHost, stream));
        if (include_query) {
            CUDA_CHECK(cudaMemcpyAsync(host + statistics.key_bytes() + shard.first_layer * query_stride,
                shard.query_sums.data, shard.query_sums.bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }
    device.synchronize();
}
} // namespace

void ProgramImpl::prepare_memory_statistics(SequenceState& sequence, std::uint32_t query_begin,
                                            std::uint32_t query_end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics) {
        const auto& config = parameters.model.config().text;
        state.statistics = std::make_unique<kvmem::RawKvStore>(kvmem::RawKvStoreConfig{
            .n_layer = config.full_attention_layers,
            .n_embd_k = memory_statistics->key_width,
            .block_tokens = 64});
        kvmem::KvMemStoreConfig selection;
        selection.lrfu = kvmem_options.lrfu;
        selection.lrfu_lambda = kvmem_options.lrfu_lambda;
        selection.lrfu_top_fraction = kvmem_options.lrfu_top_fraction;
        // ActKV pillar-3 (CalAttn). These used to be left at the library defaults, which pinned the
        // ninfer backend to the weakest scorer (MeanK) even though the shared core ships PerToken
        // and SubBlockMeanK with the MaxSim-style sub-block reduction. Selecting them here is the
        // whole "wire, do not rewrite" step of the pillar.
        selection.retrieval_method = static_cast<kvmem::KvMemRetrievalMethod>(
            static_cast<std::uint8_t>(kvmem_options.retrieval_method));
        selection.n_subblocks = kvmem_options.subblocks;
        selection.subblock_reduce = static_cast<kvmem::KvMemSubblockReduce>(
            static_cast<std::uint8_t>(kvmem_options.subblock_reduce));
        selection.prototype_mode = static_cast<kvmem::KvMemPrototypeMode>(
            static_cast<std::uint8_t>(kvmem_options.prototype_mode));
        selection.block_tokens = 64;
        selection.select_budget = kvmem_options.selected_tokens;
        // ActKV pillar-2: B_upper is conceptually separate from B. The device pool is allocated for
        // B and that never changes here; the confidence monitor may only widen the SEMANTIC budget,
        // and only up to this ceiling. A ceiling equal to B leaves the adaptive path inert.
        selection.select_budget_upper =
            std::min(kvmem_options.selected_tokens,
                     kvmem_options.conf_upper_tokens != 0 ? kvmem_options.conf_upper_tokens
                                                          : kvmem_options.selected_tokens);
        selection.sink_blocks = kvmem_options.sink_blocks();
        selection.recent_blocks = kvmem_options.recent_blocks();
        state.selector = std::make_unique<kvmem::KvMemStore>(selection);
        if (kvmem_options.verify_transfers) {
            // ActKV pillar-3 mechanism evidence: report what the selector ACTUALLY got. A flag that
            // parses but never reaches the selector is indistinguishable from the outside, which is
            // exactly how the reference implementation's confidence module stayed dead for months.
            std::fprintf(stderr,
                "KVMEM_SCORER method=%u subblocks=%u reduce=%u prototype=%u budget=%u upper=%u "
                "lrfu=%d lambda=%.4g top_frac=%.4g conf=%d\n",
                static_cast<unsigned>(kvmem_options.retrieval_method), kvmem_options.subblocks,
                static_cast<unsigned>(kvmem_options.subblock_reduce),
                static_cast<unsigned>(kvmem_options.prototype_mode), selection.select_budget,
                selection.select_budget_upper, int(kvmem_options.lrfu), kvmem_options.lrfu_lambda,
                kvmem_options.lrfu_top_fraction, int(kvmem_options.conf_budget));
        }
        if (kvmem_options.conf_budget) {
            // Sec. 6.1: the aggregation window and the round cooldown are what make the raw trend
            // usable. These are conservative starting values for this machine, not the paper's
            // numbers transplanted from other hardware.
            state.confidence = std::make_unique<kvmem::KvMemConfidenceMonitor>(
                kvmem::KvMemConfidenceConfig{
                    .pool_size = kvmem_options.conf_pool,
                    .stride = kvmem_options.conf_stride,
                    .bottom_frac = kvmem_options.conf_bottom_frac,
                    .min_pool = kvmem_options.conf_min_pool,
                    .cooldown_rounds = kvmem_options.conf_cooldown_rounds});
            state.runtime_budget_tokens =
                std::min(selection.select_budget_upper,
                         kvmem_options.conf_init_tokens != 0 ? kvmem_options.conf_init_tokens
                                                             : kvmem_options.selected_tokens);
            if (state.runtime_budget_tokens < kvmem_options.selected_tokens) {
                state.selector->set_runtime_select_budget(state.runtime_budget_tokens);
            }
        } else {
            state.runtime_budget_tokens = kvmem_options.selected_tokens;
        }
        state.selector->begin_retrieval_round();
        state.query_sum.assign(memory_statistics->query_elements(), 0.0F);
        state.query_begin = query_begin;
        state.query_end = query_end;
    } else if (kvmem_options.lrfu && state.selector &&
               state.selector->retrieval_checkpoint().committed) {
        // ActKV round guard. This function runs once per prefill STEP (prefill.cpp:1242), while
        // the branch above runs only when the lane has no statistics yet. A COMMITTED round
        // reaching this point therefore means the previous request finished and this one never
        // passed through restore_memory_history() - the only other call site that opens a
        // keep-score round (kvmem_window.cpp:1152). LrfuHistory::preview() early-returns while
        // the round is committed, so without this the keep scores would freeze for the rest of
        // the session and every pressure reselection would rank by the PREVIOUS round's score,
        // while pick_topk_blocks kept reporting success. That is the reference implementation's
        // A9 failure shape: the first rounds look right, then the accumulated history locks the
        // selector down. Re-opening loses nothing: begin_round() advances the round id and
        // clears the committed flag, it does not clear the accumulated scores.
        state.selector->begin_retrieval_round();
        if (kvmem_options.verify_transfers) {
            std::fprintf(stderr,
                "KVMEM_LRFU_REOPEN round=%llu (a committed round reached a new request)\n",
                static_cast<unsigned long long>(state.selector->retrieval_checkpoint().round));
        }
    }
    if (state.query_begin != query_begin || state.query_end != query_end) {
        throw std::logic_error("KVMem query range changed inside one prefill");
    }
    memory_statistics_ranges = {0, static_cast<std::int32_t>(capacity),
        static_cast<std::int32_t>(query_begin), static_cast<std::int32_t>(query_end)};
    for (std::size_t rank = 0; rank < memory_statistics->ranks.size(); ++rank) {
        const auto& shard = memory_statistics->ranks[rank];
        if (shard.ranges.data == nullptr) { continue; }
        DeviceBinding binding(device.rank(rank).device);
        CUDA_CHECK(cudaMemcpyAsync(shard.ranges.data, memory_statistics_ranges.data(),
            sizeof(memory_statistics_ranges), cudaMemcpyHostToDevice, device.stream_for_rank(rank)));
    }
}

void ProgramImpl::commit_memory_statistics(SequenceState& sequence, std::uint32_t begin,
                                           std::uint32_t end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics || begin != state.statistics_frontier || end <= begin ||
        end - begin > prefill_chunk) {
        throw std::logic_error("KVMem statistics require exactly one append at the evaluated frontier");
    }
    const auto& buffers = *memory_statistics;
    auto* host = static_cast<std::byte*>(memory_statistics_host->data());
    read_statistics(device, buffers, host, true);
    const auto width = buffers.key_width;
    const auto layer_stride = static_cast<std::size_t>(width) * buffers.columns;
    const auto* keys = reinterpret_cast<const float*>(host);
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto at = begin; at < end;) {
            const auto count = std::min(end - at, 64 - at % 64);
            const auto bucket = at / 64 - begin / 64;
            const auto* sum = keys + layer * layer_stride + bucket * width;
            for (std::uint32_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(sum[channel])) { throw std::runtime_error("Nonfinite KVMem key sum"); }
            }
            state.statistics->write_layer_mean_sum(at, count, layer, sum);
            at += count;
        }
    }
    const auto query_first = std::max(begin, state.query_begin);
    const auto query_last = std::min(end, state.query_end);
    if (!state.query_replayed && query_last > query_first) {
        const auto* query = reinterpret_cast<const float*>(host + buffers.key_bytes());
        for (std::size_t i = 0; i < state.query_sum.size(); ++i) {
            if (!std::isfinite(query[i])) { throw std::runtime_error("Nonfinite KVMem query sum"); }
            state.query_sum[i] += query[i];
        }
        state.query_tokens += query_last - query_first;
    }
    if (state.selector->total_tokens() != begin) {
        throw std::logic_error("KVMem block catalog lost its statistics frontier");
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
    if (kvmem_options.verify_transfers && (end == state.query_end || end % 64 == 0)) {
        std::fprintf(stderr, "KVMEM_STATISTICS frontier=%u query_tokens=%u query_begin=%u query_end=%u layers=%u mean_bytes=%zu\n",
            end, state.query_tokens, state.query_begin, state.query_end,
            state.statistics->config().n_layer, state.statistics->allocated_bytes());
    }
}
void ProgramImpl::read_memory_candidates() {
    if (!memory_candidate_statistics) return;
    read_statistics(device, *memory_candidate_statistics, memory_candidate_host->data(), false);
}

void ProgramImpl::commit_memory_candidates(SequenceState& sequence, std::uint32_t begin,
                                            std::uint32_t end, std::uint32_t first_column,
                                            std::uint32_t row_width) {
    if (!memory_candidate_statistics) { return; }
    auto& state = sequence.window;
    const auto& keys = *memory_candidate_statistics;
    if (!state.statistics || !state.selector || begin != state.statistics_frontier ||
        state.selector->total_tokens() != begin || end <= begin ||
        end - begin > row_width || first_column + row_width > keys.columns) {
        throw std::logic_error("KVMem speculative statistics have an invalid committed prefix");
    }
    const auto width = static_cast<std::size_t>(keys.key_width);
    const auto stride = width * keys.columns;
    const auto* values = static_cast<const float*>(memory_candidate_host->data());
    // Each represented BF16 key was copied into its own FP32 column by
    // token_sums(block_tokens=1). Rejected and output-truncated columns never
    // enter the persistent mean-K store, including a partial logical page.
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto position = begin; position < end; ++position) {
            const auto* value = values + layer * stride + (first_column + position - begin) * width;
            for (std::size_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(value[channel])) {
                    throw std::runtime_error("Nonfinite committed KVMem candidate key");
                }
            }
            state.statistics->write_layer_mean_sum(position, 1, layer, value);
        }
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
}
void ProgramImpl::observe_confidence(SequenceState& sequence, std::size_t logprob_row,
                                     std::size_t rows_per_lane) {
    if (!kvmem_options.conf_budget || rows_per_lane == 0 || !logprobs_host) { return; }
    auto& monitor = sequence.window.confidence;
    if (!monitor) { return; }
    const auto capacity = static_cast<std::size_t>(io.logprob_ids.ne[1]);
    const auto* ids     = static_cast<const TokenId*>(logprobs_host->data());
    const auto* values  = reinterpret_cast<const float*>(ids + capacity * kMaximumTokenLogprobs);
    for (std::size_t at = 0; at < rows_per_lane; ++at) {
        const std::size_t slot = (logprob_row * rows_per_lane + at) * kMaximumTokenLogprobs;
        double accumulated = 0.0;
        std::size_t used   = 0;
        for (std::size_t k = 0; k < kMaximumTokenLogprobs; ++k) {
            const float value = values[slot + k];
            if (!std::isfinite(value)) { continue; }
            accumulated += value;
            ++used;
        }
        if (used == 0) { continue; }
        // T_i = -(1/k) * sum of the k largest log probabilities (Sec. 6.1). The gather already
        // holds log P, so neither raw logits nor an LSE are needed; a constant temperature scales
        // every T by the same factor and therefore cannot flip the sign of the trend.
        monitor->observe(-accumulated / static_cast<double>(used));
    }
}

void ProgramImpl::apply_confidence_budget(SequenceState& sequence) {
    if (!kvmem_options.conf_budget) { return; }
    auto& window = sequence.window;
    if (!window.confidence || !window.selector) { return; }
    if (kvmem_options.verify_transfers) {
        // ActKV pillar-2 mechanism evidence. Without this line a monitor that never became ready
        // (wiring bug) and one whose trend simply stayed non-negative (working as intended) produce
        // exactly the same log, and "no growth" cannot be told apart from "never ran".
        std::fprintf(stderr, "KVMEM_CONF observed=%u pooled=%u ready=%d slope=%.6g budget=%u/%u\n",
                     window.confidence->observed(), window.confidence->pooled(),
                     window.confidence->ready() ? 1 : 0, window.confidence->slope(),
                     window.selector->select_budget_tokens(), kvmem_options.selected_tokens);
    }
    if (!window.confidence->round_boundary()) { return; }
    const std::uint32_t ceiling =
        std::min(kvmem_options.selected_tokens,
                 kvmem_options.conf_upper_tokens != 0 ? kvmem_options.conf_upper_tokens
                                                      : kvmem_options.selected_tokens);
    const std::uint32_t current = window.selector->select_budget_tokens();
    const std::uint64_t grown =
        static_cast<std::uint64_t>(current) * kvmem_options.conf_scale_up;
    std::uint32_t next = static_cast<std::uint32_t>(std::min<std::uint64_t>(ceiling, grown));
    next               = (next / 64) * 64;
    if (next <= current) { return; }
    // Sec. 6.2 is "only ever raise". The device pool was allocated for B, so widening the semantic
    // budget moves the visible set inside capacity that already exists.
    window.selector->set_runtime_select_budget(next);
    window.runtime_budget_tokens = next;
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_CONF_BUDGET %u -> %u (slope %.6g, pooled %u)\n", current, next,
                     window.confidence->slope(), window.confidence->pooled());
    }
}
} // namespace ninfer::models::qwen3_5::detail
