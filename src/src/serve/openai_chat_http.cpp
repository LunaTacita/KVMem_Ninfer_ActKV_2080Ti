#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/stream_sessions.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    ModelRegistry::Lease lease;
    try {
        const RequestLimits limits       = request_limits(options_);
        const auto body                  = parse_json_body(req);
        request                          = parse_chat_completion_request(body, limits);
        request.generation.ngram_session = resolve_ngram_session(req, body, options_);
        lease                            = lease_model(request.model, req);
        request.model                    = lease.model();
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService* const service = &lease.service();

    // Resumable stream opt-in: the built-in WebUI tags a streaming POST with the identity it will
    // replay under (X-Conversation-Id). The session keeps producing after this connection drops;
    // untagged requests keep the old behaviour, where a disconnect cancels generation.
    std::string conv_id;
    std::shared_ptr<StreamSession> session;
    if (request.stream) {
        conv_id = req.get_header_value("X-Conversation-Id");
        if (!conv_id.empty() && conv_id.size() <= 240) {
            session = stream_sessions_.open(conv_id, request.model);
        }
    }
    // Cancellation for the preparation phase: a tagged stream must survive the client going away
    // (a reload during a slow model load or prompt rendering has to be resumable), while an
    // untagged request keeps the client's disconnect as its cancel signal.
    const std::function<bool()> prepare_cancelled =
        session ? std::function<bool()>([session] { return session->should_cancel(); })
                : std::function<bool()>([&req] { return client_disconnected(req); });

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    try {
        const ninfer::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
        };
        prepared = service->prepare(request.generation,
                                    request.stream ? GenerationConsumerMode::Streaming
                                                   : GenerationConsumerMode::Aggregate,
                                    observation, prepare_cancelled);
        prepared.model_hold = std::make_shared<ModelRegistry::Lease>(std::move(lease));
    } catch (const ApiException& exception) {
        if (session) { stream_sessions_.erase(conv_id); }
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        if (session) { stream_sessions_.erase(conv_id); }
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const RequestFailure failure =
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what());
            lifecycle->failure(failure);
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_ngram_generation_header(res, outcome.metrics.ngram_archive);
            set_owned_json_content(res, make_chat_completion_response(
                                       identity, outcome,
                                       request.generation.reported_top_logprobs()),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage,
                                                          timings_per_token, return_progress,
                                                          options_.usage_chunk_choice,
                                                          request.generation.reported_top_logprobs());

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, service, stream, encoder, lifecycle, session, conv_id, return_progress,
             timings_per_token](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                bool client_gone = false;
                // Every SSE payload also lands in the replay buffer, so a later GET /v1/stream can
                // repeat exactly these bytes. A dead socket stops the socket writes but not the
                // generation: that is what makes a reload resume instead of restart.
                const auto emit_text = [&](std::string_view payload) {
                    if (session) { session->append(payload); }
                    if (client_gone) { return; }
                    try {
                        transport.write(payload);
                        // This connection is a live consumer while it accepts bytes; the idle
                        // timeout only fires for an abandoned generation, not a slow prefill.
                        if (session) { session->touch_consumer(); }
                    } catch (const ClientDisconnected&) {
                        client_gone = true;
                    }
                };
                const auto emit_list = [&](const std::vector<std::string>& items) {
                    for (const std::string& item : items) { emit_text(item); }
                };
                const auto send_error = [&](const ApiError& error) {
                    try {
                        emit_text(sse_error_event(error));
                        sink.done();
                        return true;
                    } catch (const std::exception&) {
                        return false;
                    }
                };
                try {
                    emit_text(encoder->start());
                } catch (const ClientDisconnected&) {
                    client_gone = true;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    if (session) {
                        stream_sessions_.erase(conv_id);
                    }
                    return send_error(error);
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninfer::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) { emit_text(encoder->initial_prompt_progress()); }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const ninfer::PromptProgress& progress) {
                            emit_text(encoder->prompt_progress(progress));
                        };
                    }
                    if (timings_per_token) {
                        output.on_timing = [&](const ninfer::GenerationTimingObservation& timing) {
                            encoder->note_timing(timing);
                        };
                    }
                    output.on_content = [&](const std::string& text,
                                            std::span<const ninfer::TokenLogprob> logprobs) {
                        emit_text(encoder->content_delta(text, logprobs));
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        emit_text(encoder->reasoning_delta(text));
                    };
                    if (session) {
                        output.is_cancelled = [session, &client_gone] {
                            // While this POST connection accepts bytes it counts as the session's
                            // consumer, so a long prefill is never mistaken for an abandoned one.
                            // After a drop the replay route refreshes this when the page returns.
                            if (!client_gone) { session->touch_consumer(); }
                            return session->should_cancel();
                        };
                    } else {
                        output.is_cancelled = [&transport] { return transport.poll(); };
                    }

                    outcome = service->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    // Only reachable without a session (a tagged stream swallows disconnects in
                    // emit_text); the untagged path keeps the old cancel-on-disconnect semantics.
                    if (session) { session->finish(); }
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    if (session) { session->finish(); }
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    if (session) { session->finish(); }
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    if (session) { session->finish(); }
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                lifecycle->done(outcome);
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                    if (auto comment = ngram_generation_comment(outcome.metrics.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    if (session) { session->finish(); }
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }
                emit_list(terminal);
                if (session) { session->finish(); }
                sink.done();
                return true;
            },
            [stream, session, lifecycle](bool successful) {
                // Without a session the connection owns the generation: closing it cancels, as
                // before. With one, the session owns the remaining output and a replay route may
                // still attach, so the release of this connection must not cancel anything.
                if (session) { return; }
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        if (session) { stream_sessions_.erase(conv_id); }
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
