find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(models/qwen4_exp/tests.cmake)
ninfer_add_test(ninfer_tool_constraints_test STANDALONE
  SOURCES models/qwen3_5/test_tool_constraints.cpp LIBRARIES ninfer_serve ninfer_model_runtime ninfer_text)
ninfer_stage_runtime_dlls(ninfer_tool_constraints_test)
include(ops/tests.cmake)

# ---------------------------------------------------------------------------
# sm75 local probe configuration.
#
# README.md's real-model examples name a 24 GB-class IQ3S vision+MTP artifact
# (model-vision-mtp.ninfer) that does not exist on this host, and they write
# their scratch files under C:/work. Every real-model probe registered below
# therefore runs against the DELIVERED Bonsai artifact and writes only under
# this stack's own Cache root, so nothing lands outside the stack directory.
# (The upstream artifact also leaves the CTest registrations without TEST_ARGS,
# which is why these probes reported their usage text as a failure.)
#
# Keeping the paths in CMake variables means moving the model or the cache does
# not mean editing thirteen registrations. Override with -DNINFER_PROBE_MODEL=
# or -DNINFER_PROBE_ROOT= on the configure line.
# ---------------------------------------------------------------------------
set(NINFER_PROBE_MODEL "${CMAKE_BINARY_DIR}/probe/model/Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer"
    CACHE FILEPATH "Artifact the tests/experimental real-model probes run against")
set(NINFER_PROBE_ROOT "${CMAKE_BINARY_DIR}/probe/tests"
    CACHE PATH "Scratch root for the tests/experimental real-model probes")
set(NINFER_PROBE_MEDIA "${NINFER_PROBE_ROOT}/media")
foreach(_ninfer_probe_dir "${NINFER_PROBE_ROOT}" "${NINFER_PROBE_ROOT}/out"
                           "${NINFER_PROBE_ROOT}/disk" "${NINFER_PROBE_MEDIA}")
  file(MAKE_DIRECTORY "${_ninfer_probe_dir}")
endforeach()

ninfer_finalize_bundle(ninfer_tests)

# Reuse only the objects needed to validate the native memory-policy contracts and
# the C++20 unique-string compatibility fix; do not link the full numerical suite.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_memory_policy_tests_PROGRAMS
  ninfer_cli_options_test ninfer_serve_options_test ninfer_kv_capacity_test
  ninfer_context_cache_defaults_test ninfer_unique_strings_test)
ninfer_finalize_bundle(ninfer_memory_policy_tests)
ninfer_stage_runtime_dlls(ninfer_memory_policy_tests)

# Exit code 77 is a skip for every test (tests/cuda_availability.h). Without a usable CUDA device --
# no GPU, no driver, or only the toolkit's stub driver, as on a CI runner -- a test that has no check
# of its own fails on its first CUDA call with one of these runtime messages; report that as skipped
# too, so such a box still runs and judges every host test. None of the messages can occur where a
# device is usable.
get_property(_ninfer_all_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_ninfer_all_tests} PROPERTIES SKIP_RETURN_CODE 77 SKIP_REGULAR_EXPRESSION
  "CUDA driver is a stub library;no CUDA-capable device is detected;CUDA driver version is insufficient for CUDA runtime version")
# Focused physical-pipeline qualification, reusing the existing program objects.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_pipeline_tests_PROGRAMS
  ninfer_multi_gpu_test ninfer_kv_capacity_test ninfer_serve_options_test
  ninfer_qwen3_5_stages_real_test)
set_property(GLOBAL APPEND PROPERTY NINFER_BUNDLE_ninfer_pipeline_tests_PROGRAMS
  ninfer_qwen3_5_loading_test)
set_property(GLOBAL APPEND PROPERTY NINFER_BUNDLE_ninfer_pipeline_tests_PROGRAMS
  ninfer_linear_gguf_test)
ninfer_finalize_bundle(ninfer_pipeline_tests)
ninfer_stage_runtime_dlls(ninfer_pipeline_tests)

# Every test executable shares one per-config runtime directory; stage the FFmpeg/curl DLL closure
# there once so they resolve their imports (the Windows loader only searches the executable's own
# directory, the system directories and PATH).
ninfer_stage_test_runtime_dlls(ninfer_tests)

# P1 feasibility probes: opt-in targets in the isolated checkout, no production CLI.
add_executable(ninfer_p1_engine experimental/p1_engine.cpp)
ninfer_test_includes(ninfer_p1_engine)
target_link_libraries(ninfer_p1_engine PRIVATE ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p1_engine)
ninfer_add_test(ninfer_p1_sparse STANDALONE
  SOURCES experimental/p1_sparse.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_p1_sparse)
# The argument-less sweep is four KV formats x three page sets x two offsets x twenty widths =
# 480 cases and needs about two minutes on this host; CTest's default budget kills it at 60 s,
# which reads as a Timeout defect. Raise the budget rather than shrink the coverage.
set_tests_properties(ninfer_p1_sparse PROPERTIES TIMEOUT 600)
ninfer_add_test(ninfer_p1_payload STANDALONE
  SOURCES experimental/p1_payload.cpp LIBRARIES ninfer_core)
ninfer_add_test(ninfer_p1_options STANDALONE
  SOURCES experimental/p1_options.cpp LIBRARIES ninfer::engine ninfer_core ninfer::json)
ninfer_stage_runtime_dlls(ninfer_p1_options)
ninfer_add_test(ninfer_p1_rope STANDALONE
  SOURCES experimental/p1_rope.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_p1_rope)

# The query provenance tests are also runnable without linking unrelated GPU test families.
ninfer_add_test(ninfer_memory_frontend STANDALONE NEEDS_SOURCE_DIR
  SOURCES models/qwen3_5/test_frontend.cpp LIBRARIES ninfer_engine ninfer_core ninfer::json)
ninfer_stage_runtime_dlls(ninfer_memory_frontend)
ninfer_add_test(ninfer_token_sums STANDALONE
  SOURCES experimental/token_sums.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_token_sums)
ninfer_add_test(ninfer_p4_retrieval STANDALONE
  SOURCES experimental/p4_retrieval.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" int8 eager "${NINFER_PROBE_ROOT}/out/p4_retrieval.txt")
ninfer_stage_runtime_dlls(ninfer_p4_retrieval)
ninfer_add_test(ninfer_p4_history STANDALONE
  SOURCES experimental/p4_history.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" int8 eager "${NINFER_PROBE_ROOT}/out/history.tokens" 4 input-fallback 0)
ninfer_stage_runtime_dlls(ninfer_p4_history)
# The fixture's own Host budget is an INT8-equivalent 230 MiB, and this build refuses it: at 35904
# native KV bytes per token, --max-context 16384 needs 561 MiB, so no session would ever run. The
# source keeps 230 MiB as the default for hand runs and honours NINFER_SESSION_TEST_HOST_MIB, so the
# registration asks for a budget the admission check accepts. 768 MiB leaves 1.37x headroom over the
# 561 MiB requirement, and the fixture's own assertion (history_evictions > 0) is insensitive to it:
# four archives against retained_sessions = 3 evict on the session limit, measured as 223.5 MiB
# retained with evictions 1 -> 3 at both 576 MiB and 768 MiB.
ninfer_add_test(ninfer_memory_sessions STANDALONE
  SOURCES experimental/memory_sessions.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 4 int8 eager)
set_tests_properties(ninfer_memory_sessions PROPERTIES ENVIRONMENT "NINFER_SESSION_TEST_HOST_MIB=768")
ninfer_stage_runtime_dlls(ninfer_memory_sessions)
# rk8v4, not int8: the fixture squeezes the Host budget down to ~516 MiB to force an active spill,
# and --max-context 16384 at the INT8 cost of 35904 B/token needs 561 MiB, which this box rejects
# outright ("KVMem cannot hold --max-context = 16384 tokens"). At the rk8v4 cost the same window
# fits in the same budget, so the spill path is exercised instead of refused.
ninfer_add_test(ninfer_active_kv_ssd STANDALONE
  SOURCES experimental/active_kv_ssd.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 4 rk8v4 eager)
ninfer_stage_runtime_dlls(ninfer_active_kv_ssd)
ninfer_add_test(ninfer_ssd_failure STANDALONE
  SOURCES experimental/ssd_failure.cpp LIBRARIES ninfer::engine kvmem
  TEST_ARGS "${NINFER_PROBE_MODEL}")
ninfer_stage_runtime_dlls(ninfer_ssd_failure)
# The trailing 2560 is the fixture's own documented requirement for two real 1024-token images:
# README "KVMem image concurrency" says a larger selection budget is needed and names 2560. Without
# it the Engine refuses with "selection budget cannot retain complete image regions".
ninfer_add_test(ninfer_memory_concurrency STANDALONE
  SOURCES experimental/memory_concurrency.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 3 8 1024 int8 eager adaptive
             "${NINFER_PROBE_MEDIA}/red.png" "${NINFER_PROBE_MEDIA}/blue.png" cpu 1024 2560)
ninfer_stage_runtime_dlls(ninfer_memory_concurrency)
# The lane count defaults to the upstream 8; this host cannot hold the 8-lane fixed Engine runtime
# reservation (3255459072 B required against 2955935744 B available, on every artifact), so the
# registration asks for a count this device can actually admit.
ninfer_add_test(ninfer_migration_concurrency STANDALONE
  SOURCES experimental/migration_concurrency.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 6)
ninfer_stage_runtime_dlls(ninfer_migration_concurrency)
# red/blue, not the user's ground-truth photographs: this fixture asks "what colour fills this
# image" and requires the answer to contain "red"/"blue" (memory_vision.cpp:144-145,210,225).
ninfer_add_test(ninfer_memory_vision STANDALONE
  SOURCES experimental/memory_vision.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 4 kvmem
             "${NINFER_PROBE_MEDIA}/red.png" "${NINFER_PROBE_MEDIA}/blue.png" int8 eager cpu all)
ninfer_stage_runtime_dlls(ninfer_memory_vision)
ninfer_add_test(ninfer_memory_disk STANDALONE
  SOURCES experimental/memory_disk.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 4 int8 "${NINFER_PROBE_ROOT}/disk" cold 2048 0 long graph)
ninfer_stage_runtime_dlls(ninfer_memory_disk)
# 16 MiB, not the fixture's 512 MiB default: its own history plus snapshot state does not fit in
# 16 MiB, so the cold leg actually spills and the persistence assertion can be satisfied. The
# source keeps 512 MiB as the default for hand runs.
set_tests_properties(ninfer_memory_disk PROPERTIES ENVIRONMENT "NINFER_DISK_TEST_HOST_MIB=36")
# The second registration runs the clear arm: reuse is switched off, so nothing may be restored from
# disk and nothing may be written either. It keeps the fixture's own 512 MiB Host default, which the
# history fits in comfortably. The plain "hit" arm cannot be a second process here -- the spill
# directory is documented as ephemeral (serve_options.cpp:201) and is deleted when its owner exits
# -- so the cold registration above qualifies reuse inside one process instead.
ninfer_add_test(ninfer_memory_disk_final STANDALONE
  SOURCES experimental/memory_disk.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 4 int8 "${NINFER_PROBE_ROOT}/disk" clear 2048 0 long graph)
ninfer_stage_runtime_dlls(ninfer_memory_disk_final)
# Independently linked so the preceding P6-P9 qualification keeps its binaries.
ninfer_add_test(ninfer_memory_formats STANDALONE
  SOURCES experimental/p4_history.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" rk8v4 eager "${NINFER_PROBE_ROOT}/out/formats.tokens" 2 input-fallback 0)
ninfer_stage_runtime_dlls(ninfer_memory_formats)
ninfer_add_test(ninfer_memory_ngram STANDALONE
  SOURCES experimental/memory_ngram.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" int8 eager 15 3 fixed)
ninfer_stage_runtime_dlls(ninfer_memory_ngram)
ninfer_add_test(ninfer_p4_quality STANDALONE
  SOURCES experimental/p4_quality.cpp LIBRARIES ninfer::engine
  TEST_ARGS "${NINFER_PROBE_MODEL}" 262144 "${NINFER_PROBE_ROOT}/out/quality.txt" 0 int8)
ninfer_stage_runtime_dlls(ninfer_p4_quality)

set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_kvmem_product_tests_PROGRAMS
  ninfer_serve_options_test ninfer_openai_schema_test ninfer_openai_responses_test
  ninfer_openai_responses_store_test ninfer_load_report_test ninfer_resource_manager_test)
ninfer_finalize_bundle(ninfer_kvmem_product_tests)
ninfer_stage_runtime_dlls(ninfer_kvmem_product_tests)
add_test(NAME ninfer_openai_image_detail_chat_test
  COMMAND ninfer_kvmem_product_tests ninfer_openai_schema_test --image-detail-only)
add_test(NAME ninfer_openai_image_detail_responses_test
  COMMAND ninfer_kvmem_product_tests ninfer_openai_responses_test --image-detail-only)

# Reuse only the GGUF numerical test object for the focused format qualification.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_migration_gguf_tests_PROGRAMS
  ninfer_linear_gguf_test)
ninfer_finalize_bundle(ninfer_migration_gguf_tests)
ninfer_stage_runtime_dlls(ninfer_migration_gguf_tests)

# Keep codec, Attention and MTP route qualification runnable without linking
# unrelated numerical families into the NVFP4 integration executable.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_kvmem_nvfp4_tests_PROGRAMS
  ninfer_kv_cache_append_test ninfer_softmax_attention_test
  ninfer_qwen3_5_mtp_graph_profiles_test ninfer_serve_options_test)
ninfer_finalize_bundle(ninfer_kvmem_nvfp4_tests)
ninfer_stage_runtime_dlls(ninfer_kvmem_nvfp4_tests)
