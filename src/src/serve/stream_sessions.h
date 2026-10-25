#pragma once

// Process-local replay buffers for resumable chat streams. The built-in WebUI opts in by tagging a
// streaming POST with X-Conversation-Id; from then on the server keeps producing into a session
// buffer when that connection drops, and GET /v1/stream replays the buffer from a byte offset so a
// reloaded page continues the same generation. Untagged requests keep the old behaviour: a client
// disconnect cancels generation.
//
// The session is a bounded buffer plus a condition variable, not a second generation owner: the
// POST handler thread still drives the one generation, it just stops writing to a dead socket
// instead of cancelling. Consumers (the original request and any replay route) read the same bytes.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

struct StreamSessionLimits {
    // Sessions kept at once, running or finished and waiting to be replayed.
    std::size_t max_sessions = 8;
    // Per-session replay buffer cap. Hitting it truncates the buffer; the generation keeps running
    // and the session reports truncated() so a consumer can stop instead of waiting for bytes that
    // will never arrive.
    std::size_t max_buffer_bytes = 64U << 20U;
    // A generation whose consumer has been gone this long is cancelled, and a finished session is
    // dropped once it has been unread this long. This bounds what an abandoned tab can pin.
    std::chrono::seconds idle_timeout{300};
};

struct StreamSessionInfo {
    std::string conv_id;
    std::string model;
    bool done                  = false;
    bool truncated             = false;
    std::size_t total_bytes    = 0;
    std::uint64_t started_at   = 0; // unix milliseconds
    std::uint64_t completed_at = 0;
};

class StreamSession {
public:
    // Writes one chunk to a consumer; returns false when the consumer is gone.
    using WriteFn = std::function<bool(std::string_view)>;

    StreamSession(std::string conv_id, std::string model, StreamSessionLimits limits);

    [[nodiscard]] const std::string& conv_id() const noexcept { return conv_id_; }
    [[nodiscard]] const std::string& model() const noexcept { return model_; }

    // Generation side: append the SSE bytes produced so far.
    void append(std::string_view bytes);
    // Generation side: the generation ended (normally or with a failure); no more bytes come.
    void finish();
    // Generation side: the engine's wait loop asks whether it should stop. True after a DELETE, or
    // when no consumer has read the session for idle_timeout (an abandoned generation).
    [[nodiscard]] bool should_cancel() const noexcept;
    // Generation side: records that a connection is still attached to this session (the POST that
    // drives the generation, or a replay GET). An abandoned tab therefore cancels after
    // idle_timeout instead of pinning the GPU until the output budget runs out.
    void touch_consumer() noexcept;
    // Route side: DELETE /v1/stream.
    void request_cancel() noexcept;

    // Consumer side: copy the buffer from `cursor` and block for more bytes until the generation
    // ends, the session is cancelled, or `write` reports a dead consumer. A quiet stream gets an
    // SSE comment heartbeat so an idle connection is not mistaken for a dead one.
    bool pump(const WriteFn& write, std::size_t& cursor);

    [[nodiscard]] bool done() const noexcept { return done_.load(std::memory_order_acquire); }
    [[nodiscard]] bool truncated() const noexcept { return truncated_.load(std::memory_order_acquire); }
    [[nodiscard]] std::size_t total_bytes() const;
    [[nodiscard]] std::uint64_t started_at() const noexcept { return started_at_; }
    [[nodiscard]] std::uint64_t completed_at() const noexcept {
        return completed_at_.load(std::memory_order_acquire);
    }
    // Registry housekeeping: a finished session nobody replayed is dropped after idle_timeout.
    [[nodiscard]] bool retention_expired(std::chrono::steady_clock::time_point now) const noexcept;

private:
    std::string conv_id_;
    std::string model_;
    StreamSessionLimits limits_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::string buffer_;

    std::atomic<bool> done_{false};
    std::atomic<bool> cancelled_{false};
    std::atomic<bool> truncated_{false};
    std::uint64_t started_at_ = 0; // unix milliseconds
    std::atomic<std::uint64_t> completed_at_{0};
    // steady_clock milliseconds of the moment the buffer was last read (or the session started);
    // the generation is cancelled when nothing has read it for idle_timeout.
    std::atomic<std::int64_t> last_read_ms_{0};
    std::atomic<std::int64_t> finished_steady_ms_{0};
};

class StreamSessionRegistry {
public:
    explicit StreamSessionRegistry(StreamSessionLimits limits = {});

    // Registers the session for a streaming request. An existing session with the same identity is
    // cancelled and replaced: a client must finish or delete a stream before reusing its identity.
    // Null when the registry is at its session cap; the caller then serves without replay.
    std::shared_ptr<StreamSession> open(std::string conv_id, std::string model);
    [[nodiscard]] std::shared_ptr<StreamSession> find(std::string_view conv_id);
    // The DELETE route: cancel the generation and release the buffer.
    bool erase(std::string_view conv_id);
    // POST /v1/streams/lookup: one entry per known id, in request order.
    [[nodiscard]] std::vector<StreamSessionInfo> lookup(const std::vector<std::string>& ids);
    [[nodiscard]] std::size_t size() const;

private:
    // Drops finished sessions past retention and makes room for one more session. A session that
    // is still running is never dropped implicitly.
    void prune_locked(std::chrono::steady_clock::time_point now);

    StreamSessionLimits limits_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<StreamSession>> sessions_;
};

} // namespace ninfer::serve
