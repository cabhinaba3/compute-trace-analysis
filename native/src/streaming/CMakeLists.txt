# Streaming graph sources, sketches, caches, and pluggable engines.
target_sources(_native PRIVATE
  sources/chakra_et_stream.cpp
  sources/chakra_host_stream.cpp
  sources/goal_bin_stream.cpp
  sources/goal_text_stream.cpp
  engines/engine_registry.cpp
  engines/online_agg/online_agg.cpp
  engines/frontier/frontier_engine.cpp)

# Deterministic floating-point behavior across CPUs (no FMA contraction).
set_source_files_properties(
  engines/frontier/frontier_engine.cpp
  TARGET_DIRECTORY _native
  PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
