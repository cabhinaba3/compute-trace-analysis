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
include(JaxProto)     # C++ sources generated from vendored XPlane and Perfetto schemas

# Optional GPU backends of the what-if solver (src/gpu). CTA_GPU is "auto" (use what the machine has),
# "OFF", "ON" (CUDA on Linux, Metal on macOS, an error if missing), "CUDA" or "METAL". A build without
# a toolkit is the same module with the CPU solver only.
set(CTA_GPU "auto" CACHE STRING "GPU backends: auto, ON, OFF, CUDA or METAL")
string(TOUPPER "${CTA_GPU}" _cta_gpu)
set(CTA_HAVE_CUDA OFF)
set(CTA_HAVE_METAL OFF)
if(NOT _cta_gpu STREQUAL "OFF")
  include(CheckLanguage)
  if(APPLE AND (_cta_gpu STREQUAL "AUTO" OR _cta_gpu STREQUAL "ON" OR _cta_gpu STREQUAL "METAL"))
    check_language(OBJCXX)
    if(CMAKE_OBJCXX_COMPILER)
      enable_language(OBJCXX)
      find_library(CTA_METAL_FRAMEWORK Metal)
      find_library(CTA_FOUNDATION_FRAMEWORK Foundation)
      if(CTA_METAL_FRAMEWORK AND CTA_FOUNDATION_FRAMEWORK)
        set(CTA_HAVE_METAL ON)
      endif()
    endif()
  elseif(NOT APPLE AND (_cta_gpu STREQUAL "AUTO" OR _cta_gpu STREQUAL "ON" OR _cta_gpu STREQUAL "CUDA"))
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
          set(CMAKE_CUDA_ARCHITECTURES native)
        else()
          set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90")
        endif()
      endif()
      set(CMAKE_CUDA_STANDARD 17)
      enable_language(CUDA)
      find_package(CUDAToolkit)
      if(CUDAToolkit_FOUND)
        set(CTA_HAVE_CUDA ON)
      endif()
    endif()
  endif()
  if((_cta_gpu STREQUAL "CUDA" AND NOT CTA_HAVE_CUDA) OR (_cta_gpu STREQUAL "METAL" AND NOT CTA_HAVE_METAL)
     OR (_cta_gpu STREQUAL "ON" AND NOT CTA_HAVE_CUDA AND NOT CTA_HAVE_METAL))
    message(FATAL_ERROR "CTA_GPU=${CTA_GPU}, but no CUDA toolkit (Linux) or Metal (macOS) was found")
  endif()
endif()
message(STATUS "cta: GPU backends: cuda=${CTA_HAVE_CUDA} metal=${CTA_HAVE_METAL}")

# One module. Each area under src/ adds its sources with target_sources() in
# its own CMakeLists.txt, so a new reader or generator only touches its
# directory. src/ is the include root: every #include names its file from
# there ("util/sink.hpp").
nanobind_add_module(_native NB_STATIC src/module.cpp ${CHAKRA_PROTO_SRCS} ${JAX_PROTO_SRCS})
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(_native PRIVATE ZLIB::ZLIB SQLite::SQLite3 protobuf::libprotobuf simdjson::simdjson
                                      Threads::Threads)
target_compile_options(_native PRIVATE -Wall -Wextra)
set_target_properties(_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CTA_NATIVE_OUTPUT}")

add_subdirectory(src/analysis)   # DAG solver and interval helpers
add_subdirectory(src/readers)    # trace readers
add_subdirectory(src/gpu)        # optional GPU backends of the what-if solver
add_subdirectory(src/sim)        # the discrete-event simulator
add_subdirectory(src/streaming)  # streaming graph sources, sketches, and engines
add_subdirectory(src/synth)      # synthetic trace generators
add_subdirectory(src/bindings)   # the Python-visible functions

# No fused multiply-add: what-if answers must not depend on the CPU.
set_source_files_properties(src/analysis/whatif.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
# The quantile sketches are header-only and also run from their Python bindings. arm64 fuses
# multiply-adds by default, x86 only with -march flags, so results would differ by platform.
set_source_files_properties(src/bindings/streaming.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
