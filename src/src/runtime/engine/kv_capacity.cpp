#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

void validate_curve(const SequenceCapacityCurve& curve) {
    if (curve.main_page_tokens == 0 || curve.minimum_main_page_groups == 0 ||
        curve.minimum_main_page_groups > curve.maximum_main_page_groups ||
        curve.minimum_device_reservation_bytes == 0) {
        throw std::invalid_argument("sequence capacity curve is invalid");
    }
    for (const RankCapacityCurve& rank : curve.extra_ranks) {
        if (rank.minimum_device_reservation_bytes == 0) {
            throw std::invalid_argument("sequence capacity curve has an empty device reservation");
        }
    }
    if (curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable sequence capacity curve has zero byte stride");
    }
}

std::uint32_t explicit_page_groups(const KvCapacityPolicy& policy,
                                   const SequenceCapacityCurve& curve) {
    if (policy.explicit_tokens == 0) {
        throw std::invalid_argument("explicit KV capacity must be positive");
    }
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) / curve.main_page_tokens;
    if (pages + curve.resident_main_pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages + curve.resident_main_pages);
}

// Appended to a capacity refusal so the reader gets the limit as well as the two
// byte counts (R3: name how much this machine can start instead of only failing).
//
// The curve's minimum page count IS the window a KVMem run derives from B+R, and its
// reservation is affine in the page count (fixed + pages x stride). A budget below the
// minimum therefore does NOT mean "nothing fits": it means the window itself has to
// shrink. Recovering the stride-free fixed cost makes the largest affordable page count
// a closed form, so a refusal can name the window this budget really admits.
struct AdmissibleCapacity {
    std::uint64_t pages       = 0;
    std::uint64_t fixed       = 0;
    std::uint64_t requested   = 0;
    bool          stride_free = false;
};

AdmissibleCapacity admissible_capacity(const SequenceCapacityCurve& curve, std::size_t budget,
                                       std::size_t stride) {
    AdmissibleCapacity out;
    out.requested   = curve.minimum_main_page_groups;
    out.stride_free = stride == 0;
    if (out.stride_free) {
        out.pages = budget >= curve.minimum_device_reservation_bytes
                        ? curve.minimum_main_page_groups
                        : 0;
        return out;
    }
    const std::uint64_t covered =
        static_cast<std::uint64_t>(curve.minimum_main_page_groups) * stride;
    out.fixed = curve.minimum_device_reservation_bytes > covered
                    ? curve.minimum_device_reservation_bytes - covered
                    : 0;
    out.pages = budget > out.fixed ? (budget - out.fixed) / stride : 0;
    return out;
}

std::string admissible_note(const SequenceCapacityCurve& curve, std::size_t budget) {
    const AdmissibleCapacity fit =
        admissible_capacity(curve, budget, curve.bytes_per_additional_main_page_group);
    if (fit.stride_free) {
        return "; this budget admits at most the minimum Main-KV capacity of " +
               std::to_string(curve.minimum_main_page_groups) + " page group(s)";
    }
    if (fit.pages == 0) {
        return "; this budget cannot hold even the " + std::to_string(fit.fixed) +
               " bytes of Engine runtime reservation that do not scale with the KV capacity";
    }
    return "; the largest Main-KV capacity this budget admits is " + std::to_string(fit.pages) +
           " page group(s) = " + std::to_string(fit.pages * curve.main_page_tokens) +
           " KV tokens, while this request asks for " + std::to_string(fit.requested) +
           " page group(s) = " + std::to_string(fit.requested * curve.main_page_tokens) +
           " KV tokens, so the window must shrink";
}

std::string admissible_rank_note(const SequenceCapacityCurve& curve, std::size_t rank,
                                 std::size_t budget) {
    const RankCapacityCurve& held = curve.extra_ranks[rank];
    if (held.bytes_per_additional_main_page_group == 0) {
        return budget >= held.minimum_device_reservation_bytes
                   ? "; this device holds no KV, so it does not limit the page count"
                   : "; this budget cannot hold even the minimum Engine runtime reservation "
                     "on this device";
    }
    const std::uint64_t covered =
        static_cast<std::uint64_t>(curve.minimum_main_page_groups) *
        held.bytes_per_additional_main_page_group;
    const std::uint64_t fixed = held.minimum_device_reservation_bytes > covered
                                    ? held.minimum_device_reservation_bytes - covered
                                    : 0;
    if (budget <= fixed) {
        return "; this budget cannot hold even the " + std::to_string(fixed) +
               " bytes of Engine runtime reservation that do not scale with the KV capacity";
    }
    const std::uint64_t pages = (budget - fixed) / held.bytes_per_additional_main_page_group;
    return "; the largest Main-KV capacity this device admits is " + std::to_string(pages) +
           " page group(s) = " + std::to_string(pages * curve.main_page_tokens) +
           " KV tokens, while this request asks for " +
           std::to_string(curve.minimum_main_page_groups) + " page group(s)";
}

} // namespace

std::size_t SequenceCapacityCurve::reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(minimum_device_reservation_bytes,
                       checked_mul(additional, bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::size_t SequenceCapacityCurve::extra_rank_reservation_bytes(
    std::size_t extra_rank, std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (extra_rank >= extra_ranks.size()) {
        throw std::out_of_range("capacity curve has no such device");
    }
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const RankCapacityCurve& rank = extra_ranks[extra_rank];
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(rank.minimum_device_reservation_bytes,
                       checked_mul(additional, rank.bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::uint32_t SequenceCapacityCurve::resolved_tokens(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::uint64_t tokens = static_cast<std::uint64_t>(main_page_groups) * main_page_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("resolved KV token capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                         const SequenceCapacityCurve& curve,
                                         std::size_t available_runtime_bytes,
                                         std::span<const std::size_t> extra_rank_available_bytes) {
    validate_curve(curve);
    if (extra_rank_available_bytes.size() != curve.extra_ranks.size()) {
        throw std::invalid_argument("free memory was not given for every device in the plan");
    }
    const auto device_name = [](std::size_t rank) { return "device " + std::to_string(rank); };

    std::uint32_t pages         = curve.minimum_main_page_groups;
    std::size_t capacity_budget = available_runtime_bytes;
    std::size_t binding_rank_of_pages = 0;
    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
        pages = explicit_page_groups(policy, curve);
        break;
    case KvCapacityMode::Automatic:
        if (available_runtime_bytes < policy.automatic_headroom_bytes) {
            throw std::invalid_argument(
                "automatic KV headroom requires " +
                std::to_string(policy.automatic_headroom_bytes) + " bytes, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        capacity_budget -= policy.automatic_headroom_bytes;
        if (capacity_budget < curve.minimum_device_reservation_bytes) {
            throw std::invalid_argument(
                "minimum Engine runtime reservation requires " +
                std::to_string(curve.minimum_device_reservation_bytes) + " bytes in addition to " +
                std::to_string(policy.automatic_headroom_bytes) +
                " bytes of automatic headroom, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights" +
                (curve.extra_ranks.empty() ? std::string() : " on " + device_name(0)) +
                admissible_note(curve, capacity_budget));
        }
        if (curve.minimum_main_page_groups < curve.maximum_main_page_groups) {
            // The capacity is the smallest any device allows. Start from the primary device.
            const std::size_t additional =
                (capacity_budget - curve.minimum_device_reservation_bytes) /
                curve.bytes_per_additional_main_page_group;
            std::uint64_t candidate =
                static_cast<std::uint64_t>(curve.minimum_main_page_groups) + additional;
            for (std::size_t index = 0; index < curve.extra_ranks.size(); ++index) {
                const RankCapacityCurve& rank = curve.extra_ranks[index];
                const std::size_t available   = extra_rank_available_bytes[index];
                if (available < policy.automatic_headroom_bytes ||
                    available - policy.automatic_headroom_bytes <
                        rank.minimum_device_reservation_bytes) {
                    throw std::invalid_argument(
                        "minimum Engine runtime reservation requires " +
                        std::to_string(rank.minimum_device_reservation_bytes) +
                        " bytes in addition to " +
                        std::to_string(policy.automatic_headroom_bytes) +
                        " bytes of automatic headroom, but only " + std::to_string(available) +
                        " bytes are available after weights on " + device_name(index + 1));
                }
                // A device that holds no KV does not limit how many pages fit.
                if (rank.bytes_per_additional_main_page_group == 0) { continue; }
                const std::size_t rank_additional =
                    (available - policy.automatic_headroom_bytes -
                     rank.minimum_device_reservation_bytes) /
                    rank.bytes_per_additional_main_page_group;
                const std::uint64_t rank_candidate =
                    static_cast<std::uint64_t>(curve.minimum_main_page_groups) + rank_additional;
                if (rank_candidate < candidate) {
                    candidate            = rank_candidate;
                    binding_rank_of_pages = index + 1;
                }
            }
            pages = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(candidate, curve.maximum_main_page_groups));
        }
        break;
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    const std::size_t reservation = curve.reservation_bytes(pages);
    if (reservation > capacity_budget) {
        throw std::invalid_argument("requested Engine runtime reservation requires " +
                                    std::to_string(reservation) + " bytes, but only " +
                                    std::to_string(capacity_budget) +
                                    " bytes are available for runtime capacity" +
                                    (curve.extra_ranks.empty() ? std::string()
                                                               : " on " + device_name(0)) +
                                    admissible_note(curve, capacity_budget) + ".");
    }
    std::vector<std::size_t> extra_reservations;
    for (std::size_t index = 0; index < curve.extra_ranks.size(); ++index) {
        const std::size_t need = curve.extra_rank_reservation_bytes(index, pages);
        // Explicit capacity is not reduced by headroom on any device; automatic already was.
        const std::size_t budget =
            policy.mode == KvCapacityMode::Automatic
                ? extra_rank_available_bytes[index] - policy.automatic_headroom_bytes
                : extra_rank_available_bytes[index];
        if (need > budget) {
            throw std::invalid_argument("requested Engine runtime reservation requires " +
                                        std::to_string(need) + " bytes, but only " +
                                        std::to_string(budget) +
                                        " bytes are available for runtime capacity on " +
                                        device_name(index + 1) +
                                        admissible_rank_note(curve, index, budget) + ".");
        }
        extra_reservations.push_back(need);
    }

    return KvCapacityResolution{
        .mode                                 = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                      = curve.resolved_tokens(pages),
        .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes            = reservation,
        .extra_rank_reservation_bytes         = std::move(extra_reservations),
        .binding_rank                         = binding_rank_of_pages,
        .available_after_weights_bytes        = available_runtime_bytes,
        .automatic_headroom_bytes             = policy.automatic_headroom_bytes,
        .planned_slack_bytes                  = available_runtime_bytes - reservation,
    };
}

KvCapacityResolution resolve_mixed_kv_capacity(const KvCapacityPolicy& policy,
                                               const SequenceCapacityCurve& curve,
                                               std::uint32_t max_context,
                                               std::uint32_t max_concurrency,
                                               std::size_t available_runtime_bytes) {
    validate_curve(curve);
    if (!curve.extra_ranks.empty()) {
        throw std::invalid_argument("mixed CUDA memory supports only one device");
    }
    if (max_context == 0 || max_concurrency == 0 || max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("mixed KV capacity needs positive max_context and concurrency in [1,8]");
    }
    if (policy.automatic_headroom_bytes != 0) {
        throw std::invalid_argument("mixed CUDA memory does not support automatic KV headroom; use auto with zero headroom");
    }
    std::uint32_t pages = 0;
    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.explicit_tokens < max_context) {
            throw std::invalid_argument("mixed explicit KV capacity must be at least max_context");
        }
        pages = explicit_page_groups(policy, curve);
        break;
    case KvCapacityMode::Automatic: {
        if (policy.explicit_tokens != 0) {
            throw std::invalid_argument("mixed automatic KV capacity must not carry explicit tokens");
        }
        const std::uint64_t window_pages =
            1ULL + (static_cast<std::uint64_t>(max_context) - 1ULL) / curve.main_page_tokens;
        // The operands are bounded uint32 and concurrency <= 8, so the multiplication fits uint64.
        // Bound in page space before converting back to the public uint32 token capacity.
        const std::uint64_t requested_pages = window_pages * max_concurrency;
        pages = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(requested_pages, curve.maximum_main_page_groups));
        break;
    }
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }
    const std::size_t reservation = curve.reservation_bytes(pages);
    const std::uint32_t tokens = curve.resolved_tokens(pages);
    if (tokens < max_context) {
        throw std::invalid_argument("mixed KV capacity curve cannot hold one max_context window");
    }
    return KvCapacityResolution{
        .mode                                 = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                      = tokens,
        .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes            = reservation,
        .available_after_weights_bytes        = available_runtime_bytes,
        .automatic_headroom_bytes             = 0,
        .planned_slack_bytes                  = available_runtime_bytes > reservation
                                                   ? available_runtime_bytes - reservation : 0,
    };
}

} // namespace ninfer::runtime
