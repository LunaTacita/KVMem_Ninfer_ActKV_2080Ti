#include "serve/stream_sessions.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace ninfer::serve {
namespace {

std::uint64_t unix_milliseconds() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
}

std::int64_t monotonic_milliseconds() {
    return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
}

std::int64_t monotonic_milliseconds(std::chrono::steady_clock::time_point now) {
    return static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

} // namespace

StreamSession::StreamSession(std::string conv_id, std::string model, StreamSessionLimits limits)
    : conv_id_(std::move(conv_id)), model_(std::move(model)), limits_(limits),
      started_at_(unix_milliseconds()), last_read_ms_(monotonic_milliseconds()) {}

void StreamSession::append(std::string_view bytes) {
    if (bytes.empty()) { return; }
    {
        const std::lock_guard lock(mutex_);
        if (done_.load(std::memory_order_acquire) || truncated_.load(std::memory_order_acquire)) {
            return;
        }
        if (buffer_.size() + bytes.size() > limits_.max_buffer_bytes) {
            truncated_.store(true, std::memory_order_release);
            return;
        }
        buffer_.append(bytes);
    }
    cv_.notify_all();
}

void StreamSession::finish() {
    bool expected = false;
    if (!done_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) { return; }
    completed_at_.store(unix_milliseconds(), std::memory_order_release);
    finished_steady_ms_.store(monotonic_milliseconds(), std::memory_order_release);
    cv_.notify_all();
}

bool StreamSession::should_cancel() const noexcept {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (done()) { return false; }
    const auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(limits_.idle_timeout)
                             .count();
    const auto last = last_read_ms_.load(std::memory_order_acquire);
    return last != 0 && monotonic_milliseconds() - last > idle_ms;
}

void StreamSession::request_cancel() noexcept {
    cancelled_.store(true, std::memory_order_release);
    cv_.notify_all();
}

void StreamSession::touch_consumer() noexcept {
    last_read_ms_.store(monotonic_milliseconds(), std::memory_order_release);
}

std::size_t StreamSession::total_bytes() const {
    const std::lock_guard lock(mutex_);
    return buffer_.size();
}

bool StreamSession::retention_expired(std::chrono::steady_clock::time_point now) const noexcept {
    if (!done()) { return false; }
    const auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(limits_.idle_timeout)
                             .count();
    const auto finished = finished_steady_ms_.load(std::memory_order_acquire);
    return finished != 0 && monotonic_milliseconds(now) - finished > idle_ms;
}

bool StreamSession::pump(const WriteFn& write, std::size_t& cursor) {
    constexpr auto kHeartbeatInterval = std::chrono::seconds(5);
    auto last_write                   = std::chrono::steady_clock::now();
    for (;;) {
        std::string chunk;
        bool finished = false;
        {
            std::unique_lock lock(mutex_);
            if (cursor > buffer_.size()) { cursor = buffer_.size(); }
            if (buffer_.size() > cursor) {
                chunk.assign(buffer_, cursor, buffer_.size() - cursor);
                cursor = buffer_.size();
            } else if (done() || cancelled_.load(std::memory_order_acquire) ||
                       truncated_.load(std::memory_order_acquire)) {
                finished = true;
            } else {
                cv_.wait_for(lock, std::chrono::milliseconds(200));
            }
        }
        if (finished) { return true; }
        touch_consumer();
        if (!chunk.empty()) {
            if (!write(chunk)) { return false; }
            last_write = std::chrono::steady_clock::now();
        } else if (std::chrono::steady_clock::now() - last_write >= kHeartbeatInterval) {
            // A quiet generation (a long prefill, or a model that thinks before its first token)
            // keeps the connection visibly alive for proxies and TCP liveness checks. The client's
            // SSE reader skips comment lines, so this costs it nothing.
            if (!write(": keep-alive\n\n")) { return false; }
            last_write = std::chrono::steady_clock::now();
        }
    }
}

StreamSessionRegistry::StreamSessionRegistry(StreamSessionLimits limits) : limits_(limits) {}

void StreamSessionRegistry::prune_locked(std::chrono::steady_clock::time_point now) {
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        // A running session whose consumer is gone is dropped from the registry too; the handler
        // thread still owns the one generation and stops it on its next cancel poll.
        if (it->second->retention_expired(now) ||
            (!it->second->done() && it->second->should_cancel())) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
    // Make room for one more session by dropping the oldest finished one; a running session is
    // never dropped, so a full registry degrades to "no replay" instead of killing a generation.
    while (sessions_.size() >= limits_.max_sessions) {
        auto victim = sessions_.end();
        for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
            if (!it->second->done()) { continue; }
            if (victim == sessions_.end() ||
                it->second->started_at() < victim->second->started_at()) {
                victim = it;
            }
        }
        if (victim == sessions_.end()) { break; }
        sessions_.erase(victim);
    }
}

std::shared_ptr<StreamSession> StreamSessionRegistry::open(std::string conv_id, std::string model) {
    const std::lock_guard lock(mutex_);
    prune_locked(std::chrono::steady_clock::now());
    if (const auto existing = sessions_.find(conv_id); existing != sessions_.end()) {
        existing->second->request_cancel();
        sessions_.erase(existing);
    }
    if (sessions_.size() >= limits_.max_sessions) { return nullptr; }
    auto session = std::make_shared<StreamSession>(conv_id, std::move(model), limits_);
    sessions_.emplace(std::move(conv_id), session);
    return session;
}

std::shared_ptr<StreamSession> StreamSessionRegistry::find(std::string_view conv_id) {
    const std::lock_guard lock(mutex_);
    prune_locked(std::chrono::steady_clock::now());
    const auto it = sessions_.find(std::string(conv_id));
    return it == sessions_.end() ? nullptr : it->second;
}

bool StreamSessionRegistry::erase(std::string_view conv_id) {
    const std::lock_guard lock(mutex_);
    const auto it = sessions_.find(std::string(conv_id));
    if (it == sessions_.end()) { return false; }
    it->second->request_cancel();
    sessions_.erase(it);
    return true;
}

std::vector<StreamSessionInfo> StreamSessionRegistry::lookup(
    const std::vector<std::string>& ids) {
    const std::lock_guard lock(mutex_);
    prune_locked(std::chrono::steady_clock::now());
    std::vector<StreamSessionInfo> found;
    found.reserve(ids.size());
    for (const std::string& id : ids) {
        const auto it = sessions_.find(id);
        if (it == sessions_.end()) { continue; }
        const StreamSession& session = *it->second;
        StreamSessionInfo info;
        info.conv_id      = session.conv_id();
        info.model        = session.model();
        info.done         = session.done();
        info.truncated    = session.truncated();
        info.total_bytes  = session.total_bytes();
        info.started_at   = session.started_at();
        info.completed_at = session.completed_at();
        found.push_back(std::move(info));
    }
    return found;
}

std::size_t StreamSessionRegistry::size() const {
    const std::lock_guard lock(mutex_);
    return sessions_.size();
}

} // namespace ninfer::serve
