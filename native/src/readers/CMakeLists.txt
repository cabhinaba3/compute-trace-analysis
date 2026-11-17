# One directory per trace format.
target_sources(_native PRIVATE
  chakra/reader.cpp
  goal/reader.cpp
  goal/syntax.cpp
  kineto/reader.cpp
  kineto/chunk_parser.cpp
  kineto/merger.cpp
  kineto/source.cpp
  kineto/splitter.cpp
  nsys/reader.cpp)
