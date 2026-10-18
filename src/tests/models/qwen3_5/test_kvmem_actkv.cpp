#include "guarded_main.h"
#include "models/qwen3_5/frontend/chat_template.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

// ActKV (arXiv:2609.31395) landed in three pillars, and the engine-side wiring of pillar 1 had no
// automatic test: the previous round proved it by hand, starting a server and reading KVMEM_* log
// lines, which nothing re-runs when the code changes. Pillars 2 and 3 already have unit tests in
// the shared KVMem core (confidence_monitor_test, mean_key_retrieval_test, kvmem_store_test); this
// test covers the rule the engine applies before any of that can matter.
//
// Pillar 1 is the retrieval query. The renderer used to hand the selector the OLDEST real human
// turn, so inside an agent loop every Observation-Reason-Action round produced a byte-identical
// query: LRFU then saw a near-constant hit pattern and decayed into pure frequency, which is the
// failure ActKV exists to avoid ("frequency-dominated eviction may retain obsolete continuous
// entries"). The renderer now prefers the NEWEST assistant turn's visible region and falls back to
// the newest human turn only when no assistant turn can be represented.
//
// Rendering is pure host work -- no artifact, no CUDA, no Engine -- so a regression that
// reinstates the old rule fails here in milliseconds instead of silently degrading every long
// agent run.
namespace {

namespace fi = ninfer::models::qwen3_5::frontend;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

fi::ChatMessage message(ninfer::ChatRole role, std::string text, std::string reasoning = {},
                        std::vector<fi::ToolCall> calls = {}) {
    fi::ChatMessage out;
    out.role = role;
    out.parts.push_back(fi::ChatPart::text_part(std::move(text)));
    out.reasoning_content = std::move(reasoning);
    out.tool_calls        = std::move(calls);
    return out;
}

// The bytes memory_query selects, as text. An empty view means the render produced no query.
std::string_view query_view(const fi::RenderedChat& rendered) {
    if (!rendered.memory_query) { return {}; }
    const auto& query = *rendered.memory_query;
    return std::string_view(rendered.text.data() + query.begin, query.end - query.begin);
}

bool query_contains(const fi::RenderedChat& rendered, std::string_view needle) {
    return query_view(rendered).find(needle) != std::string_view::npos;
}

std::string read_template(const char* file) {
    std::ifstream in(std::string(NINFER_SOURCE_DIR "/tools/chat_templates/") + file,
                     std::ios::binary);
    if (!in) { return {}; }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

fi::RenderedChat render(const std::string& source, const char* name,
                        const std::vector<fi::ChatMessage>& messages) {
    const auto compiled = fi::CompiledChatTemplate::resolve(source, name);
    fi::ChatRenderOptions options;
    // The generation prompt is a per-request suffix; the query rule is about the closed history.
    options.add_generation_prompt = false;
    return compiled.render(messages, options);
}

// Arm A: a newer assistant turn exists, so the query is that turn's visible region -- and neither
// the oldest nor the middle human turn may win.
void newest_assistant_wins(const std::string& source, const char* name) {
    const std::vector<fi::ChatMessage> messages = {
        message(ninfer::ChatRole::System,
                "You are a coding agent. Use the tools to inspect files."),
        message(ninfer::ChatRole::User,
                "OLDEST_TURN_MARKER find the definition of plan_admission."),
        message(ninfer::ChatRole::Assistant, "It is defined at admission.cpp:42.",
                "The symbol is a free function in admission.cpp."),
        message(ninfer::ChatRole::User, "MIDDLE_TURN_MARKER now show me its callers."),
        message(ninfer::ChatRole::Assistant,
                "NEWEST_ACTION_MARKER it is called at core.cpp:1472.",
                "There is exactly one call site."),
    };
    const auto rendered = render(source, name, messages);
    expect(rendered.memory_query.has_value(),
           "an agent conversation must yield a retrieval query at all");
    if (!rendered.memory_query) { return; }
    expect(query_contains(rendered, "NEWEST_ACTION_MARKER"),
           "ActKV pillar 1: the query must be the newest assistant turn's visible region");
    expect(!query_contains(rendered, "OLDEST_TURN_MARKER"),
           "ActKV pillar 1: the query must not be the oldest human turn");
    expect(!query_contains(rendered, "MIDDLE_TURN_MARKER"),
           "ActKV pillar 1: a newer assistant turn outranks a newer human turn");
}

// Arm B: no assistant turn is representable, so the documented fallback -- the newest human turn --
// must apply. Without this arm, deleting the fallback entirely would go unnoticed.
void newest_user_is_the_fallback(const std::string& source, const char* name) {
    const std::vector<fi::ChatMessage> messages = {
        message(ninfer::ChatRole::System, "You are a coding agent."),
        message(ninfer::ChatRole::User, "OLD_QUESTION_MARKER what is plan_admission?"),
        message(ninfer::ChatRole::User, "NEW_QUESTION_MARKER where is it defined?"),
    };
    const auto rendered = render(source, name, messages);
    expect(rendered.memory_query.has_value(),
           "a conversation with no assistant turn must still yield a retrieval query");
    if (!rendered.memory_query) { return; }
    expect(query_contains(rendered, "NEW_QUESTION_MARKER"),
           "ActKV pillar 1: with no assistant turn the query must be the newest human turn");
    expect(!query_contains(rendered, "OLD_QUESTION_MARKER"),
           "ActKV pillar 1: the fallback must not reach back to an older human turn");
}

// Arm C: a tool loop whose newest assistant turn is a bare tool call. What the rule selects here is
// TEMPLATE DEPENDENT, and the difference is real: a template that renders reasoning for a
// still-open assistant turn gives that turn a representable region and it wins (measured:
// qwen3_8.jinja and froggeric_v22_5.jinja select "ACTION_REASONING_MARKER I need to search."),
// while a template that does not render a tool-only turn leaves it with no bytes and the rule
// correctly falls through (measured: qwen3_6.jinja selects the trailing human turn). Because the
// renderer only accepts candidate bytes that lie inside the rendered text, falling through is the
// specified behaviour rather than a defect -- so this arm always REPORTS the choice and asserts
// only where the template can represent the action.
void tool_loop_query_is_the_action(const std::string& source, const char* name,
                                   bool template_renders_tool_only_turn) {
    const std::vector<fi::ChatMessage> messages = {
        message(ninfer::ChatRole::System, "You are a coding agent."),
        message(ninfer::ChatRole::User, "TOOL_QUESTION_MARKER find every caller."),
        message(ninfer::ChatRole::Assistant, "", "ACTION_REASONING_MARKER I need to search.",
                {{"call_1", "grep", R"({"pattern": "plan_admission"})"}}),
        message(ninfer::ChatRole::Tool, "src/core.cpp:1472: plan_admission(state);"),
        message(ninfer::ChatRole::User, "TRAILING_QUESTION_MARKER thanks, continue."),
    };
    const auto rendered = render(source, name, messages);
    if (!rendered.memory_query) {
        expect(false, "a tool-loop conversation must yield a retrieval query at all");
        return;
    }
    std::cout << "kvmem_actkv tool-loop query (" << name << "): [" << query_view(rendered) << "]\n";
    if (!template_renders_tool_only_turn) { return; }
    expect(!query_contains(rendered, "TRAILING_QUESTION_MARKER"),
           "ActKV pillar 1: a later human turn must not displace the assistant action");
}

// Arm D: an assistant turn carrying prose AND a tool call. Its prose is inside the rendered text in
// every mode, so this turn must win over the trailing human turn.
void assistant_prose_with_tool_call_wins(const std::string& source, const char* name) {
    const std::vector<fi::ChatMessage> messages = {
        message(ninfer::ChatRole::System, "You are a coding agent."),
        message(ninfer::ChatRole::User, "TOOL_QUESTION_MARKER find every caller."),
        message(ninfer::ChatRole::Assistant, "ACTION_PROSE_MARKER searching now.",
                "ACTION_REASONING_MARKER let me grep.",
                {{"call_1", "grep", R"({"pattern": "plan_admission"})"}}),
        message(ninfer::ChatRole::Tool, "src/core.cpp:1472: plan_admission(state);"),
        message(ninfer::ChatRole::User, "TRAILING_QUESTION_MARKER thanks, continue."),
    };
    const auto rendered = render(source, name, messages);
    if (!rendered.memory_query) {
        expect(false, "an assistant-with-prose conversation must yield a retrieval query");
        return;
    }
    expect(query_contains(rendered, "ACTION_PROSE_MARKER") ||
               query_contains(rendered, "ACTION_REASONING_MARKER"),
           "ActKV pillar 1: an assistant turn carrying prose must be the query even when it also "
           "carries a tool call");
}

int run() {
    for (const char* file : {"qwen3_6.jinja", "qwen3_8.jinja", "froggeric_v22_5.jinja"}) {
        const std::string source = read_template(file);
        expect(!source.empty(), "the chat template must be readable under NINFER_SOURCE_DIR");
        if (source.empty()) { continue; }
        newest_assistant_wins(source, file);
        newest_user_is_the_fallback(source, file);
        // qwen3_6.jinja is the one maintained template whose tool-only assistant turn renders no
        // reasoning, so it is the only one where falling through to the human turn is expected.
        tool_loop_query_is_the_action(source, file, std::string_view(file) != "qwen3_6.jinja");
        assistant_prose_with_tool_call_wins(source, file);
    }
    if (failures == 0) {
        std::cout << "kvmem_actkv: pillar 1 query selection holds on every maintained template\n";
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
