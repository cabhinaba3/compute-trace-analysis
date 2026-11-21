# Streaming graph sources, sketches, caches, and pluggable engines.
target_sources(_native PRIVATE
  sources/chakra_et_stream.cpp
  sources/chakra_host_stream.cpp
  sources/goal_bin_stream.cpp
  sources/goal_text_stream.cpp
  cache/graph_cache.cpp
  engines/engine_registry.cpp
  engines/online_agg/online_agg.cpp
  engines/frontier/frontier_engine.cpp
  engines/gridgraph/gridgraph_engine.cpp
  engines/graphchi/graphchi_engine.cpp
  engines/xstream/xstream_engine.cpp
  engines/whatif/streaming_whatif.cpp)

# Deterministic floating-point behavior across CPUs (no FMA contraction).
set_source_files_properties(
  cache/graph_cache.cpp
  engines/frontier/frontier_engine.cpp
  engines/gridgraph/gridgraph_engine.cpp
  engines/graphchi/graphchi_engine.cpp
  engines/xstream/xstream_engine.cpp
  engines/whatif/streaming_whatif.cpp
  TARGET_DIRECTORY _native
  PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
