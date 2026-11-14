cmake_minimum_required(VERSION 3.18)
project(cta_native LANGUAGES CXX)

# Built by the root package's hatch build hook (hatch_build.py) into cta/, so
# `pip install .` produces one wheel with the extension as `cta._native`.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SQLite3 REQUIRED)

# Chakra's protobuf schema comes from the vendored submodule (third_party/chakra).
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND AND NOT protobuf_FOUND)
  find_package(Protobuf REQUIRED)  # distro packages without CMake config files
  if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf INTERFACE IMPORTED)
    target_link_libraries(protobuf::libprotobuf INTERFACE ${Protobuf_LIBRARIES})
    target_include_directories(protobuf::libprotobuf INTERFACE ${Protobuf_INCLUDE_DIRS})
  endif()
endif()
message(STATUS "cta: protobuf ${Protobuf_VERSION} (${Protobuf_DIR}) protoc=${Protobuf_PROTOC_EXECUTABLE}")
set(CHAKRA_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/chakra/schema/protobuf")
if(NOT EXISTS "${CHAKRA_PROTO_DIR}/et_def.proto")
  message(FATAL_ERROR "missing ${CHAKRA_PROTO_DIR}/et_def.proto (git submodule update --init)")
endif()
protobuf_generate_cpp(CHAKRA_PROTO_SRCS CHAKRA_PROTO_HDRS
  "${CHAKRA_PROTO_DIR}/et_def.proto" "${CHAKRA_PROTO_DIR}/storage.proto")

nanobind_add_module(_native NB_STATIC
  src/module.cpp
  src/kineto.cpp
  src/goal.cpp
  src/nsys.cpp
  src/chakra.cpp
  src/small.cpp
  ${CHAKRA_PROTO_SRCS})
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(_native PRIVATE ZLIB::ZLIB SQLite::SQLite3 protobuf::libprotobuf)
target_compile_options(_native PRIVATE -Wall -Wextra)

if(NOT CTA_NATIVE_OUTPUT)
  message(FATAL_ERROR "set -DCTA_NATIVE_OUTPUT=<dir of the cta package>")
endif()
set_target_properties(_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CTA_NATIVE_OUTPUT}")
