# Streaming graph sources, sketches, caches, and pluggable engines.
target_sources(_native PRIVATE
  sources/chakra_et_stream.cpp
  sources/chakra_host_stream.cpp
  sources/goal_bin_stream.cpp
  sources/goal_text_stream.cpp
  engines/online_agg/online_agg.cpp)
