#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// One short C<lanes> batch and its restored repeat cover lane ownership and scheduling without
// repeating the long-history stress suite. The lane count defaults to the upstream 8 and can be
// lowered on the command line: this host's device cannot hold the 8-lane fixed Engine runtime
// reservation, measured as 3255459072 B required against 2955935744 B available. That reservation
// does not scale with the KV capacity and does not change with the artifact, so neither narrowing
// the KV window nor swapping the model recovers it -- only fewer lanes do.
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) throw std::invalid_argument("artifact [concurrency 1..8]");
        const auto lanes = argc == 3 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 8U;
        if (lanes < 1 || lanes > 8) throw std::invalid_argument("concurrency must be 1..8");
        ninfer::EngineOptions o;
        ninfer::test::kvmem_test_devices(o);
        o.artifact_path = argv[1];
        o.device_profile = "off";
        o.max_context = 2048;
        o.max_concurrency = lanes;
        o.prefill_chunk = 128;
        o.kv_cache = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(lanes * 256);
        o.kvmem = {.selected_tokens = 128, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = lanes, .verify_transfers = true};
        o.context_cache.enabled = false;
        o.use_cuda_graph = true;
        o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        o.speculative.draft_tokens = 4;
        o.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
        o.speculative.ngram_draft_tokens = 15;
        ninfer::Engine engine(o);
        std::vector<ninfer::PromptInput> prompts(lanes);
        for (unsigned row = 0; row < lanes; ++row) {
            auto& p = prompts[row];
            p.context_cache.session_key = "migration-lane-" + std::to_string(row);
            p.options.enable_thinking = false;
            ninfer::ChatMessage m;
            m.role = ninfer::ChatRole::User;
            m.parts.push_back({.kind = ninfer::MessagePartKind::Text,
                .text = "List integers from " + std::to_string(100 * row) + " in order."});
            p.messages.push_back(std::move(m));
        }
        std::vector<std::vector<ninfer::TokenId>> previous(lanes);
        for (unsigned round = 0; round < 2; ++round) {
            const auto before = engine.runtime_stats();
            std::vector<std::future<ninfer::GenerationResult>> futures;
            for (const auto& p : prompts) {
                ninfer::RequestOptions request;
                request.execution.requested_output_tokens = 48;
                request.execution.sampling.temperature = 0;
                request.stop.include_model_defaults = false;
                auto handle = engine.submit(engine.prepare(p), request);
                futures.push_back(std::async(std::launch::async,
                    [h = std::move(handle)]() mutable { return h.wait(); }));
            }
            for (unsigned row = 0; row < lanes; ++row) {
                const auto r = futures[row].get();
                if (r.generated_token_ids.size() != 48 || (r.reused_prompt_tokens != 0) != (round != 0))
                    throw std::runtime_error("concurrent output or checkpoint ownership changed");
                if (round && previous[row] != r.generated_token_ids)
                    throw std::runtime_error("concurrent checkpoint replay changed greedy output");
                previous[row] = r.generated_token_ids;
                std::cout << "C" << lanes << " round=" << round << " lane=" << row
                          << " reused=" << r.reused_prompt_tokens << '\n';
            }
            const auto usage = engine.memory_summary().kvmem;
            const auto stats = engine.runtime_stats();
            if (stats.decode_row_rounds - before.decode_row_rounds <=
                stats.decode_rounds - before.decode_rounds)
                throw std::runtime_error("concurrent batch did not execute concurrent decode rows");
            if (!engine.is_available() || usage.resident_pages || usage.mtp_resident_pages ||
                usage.host_payload_bytes > o.kvmem.host_bytes)
                throw std::runtime_error("concurrent resources leaked or exceeded Host budget");
        }
        std::cout << "MIGRATION_C" << lanes << " PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MIGRATION_C" << (argc == 3 ? std::stoul(argv[2]) : 8U) << " FAIL "
                  << error.what() << '\n';
        return 1;
    }
}
