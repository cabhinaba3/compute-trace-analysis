# Finds protobuf and generates C++ for Chakra's schema, which comes from the
# vendored submodule (third_party/chakra). Sets CHAKRA_PROTO_SRCS and makes
# sure the target protobuf::libprotobuf exists.
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
