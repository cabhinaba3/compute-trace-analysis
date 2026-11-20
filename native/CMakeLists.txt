cmake_minimum_required(VERSION 3.18)
project(cta_native LANGUAGES CXX)

# Built by the root package's hatch build hook (hatch_build.py) into src/cta/,
# so `pip install .` produces one wheel with the extension as `cta._native`.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CTA_NATIVE_OUTPUT)
  message(FATAL_ERROR "set -DCTA_NATIVE_OUTPUT=<dir of the cta package>")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(simdjson CONFIG REQUIRED)
find_package(Threads REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ChakraProto)  # protobuf, and the C++ sources generated from Chakra's schema

# One module. Each area under src/ adds its sources with target_sources() in
# its own CMakeLists.txt, so a new reader or generator only touches its
# directory. src/ is the include root: every #include names its file from
# there ("util/sink.hpp").
nanobind_add_module(_native NB_STATIC src/module.cpp ${CHAKRA_PROTO_SRCS})
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(_native PRIVATE ZLIB::ZLIB SQLite::SQLite3 protobuf::libprotobuf simdjson::simdjson
                                      Threads::Threads)
target_compile_options(_native PRIVATE -Wall -Wextra)
set_target_properties(_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CTA_NATIVE_OUTPUT}")

add_subdirectory(src/analysis)  # DAG solver and interval helpers
add_subdirectory(src/readers)   # trace readers
add_subdirectory(src/synth)     # synthetic trace generators
add_subdirectory(src/bindings)  # the Python-visible functions

# No fused multiply-add: what-if answers must not depend on the CPU.
set_source_files_properties(src/analysis/whatif.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
