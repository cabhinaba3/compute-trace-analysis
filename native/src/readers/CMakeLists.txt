# One directory per trace format.
target_sources(_native PRIVATE
  chakra/reader.cpp
  goal/reader.cpp
  goal/syntax.cpp
  jax/json_reader.cpp
  jax/pftrace_reader.cpp
  jax/reader.cpp
  jax/xplane_reader.cpp
  kineto/reader.cpp
  kineto/chunk_parser.cpp
  kineto/merger.cpp
  kineto/source.cpp
  kineto/splitter.cpp
  nsys/reader.cpp)
