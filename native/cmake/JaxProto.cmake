# Generates C++ from the vendored XPlane and Perfetto protobuf schemas under
# native/proto/. Expects ChakraProto to have already located Protobuf.
set(JAX_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
if(NOT EXISTS "${JAX_PROTO_DIR}/xplane.proto")
  message(FATAL_ERROR "missing ${JAX_PROTO_DIR}/xplane.proto")
endif()
if(NOT EXISTS "${JAX_PROTO_DIR}/perfetto_trace.proto")
  message(FATAL_ERROR "missing ${JAX_PROTO_DIR}/perfetto_trace.proto")
endif()
protobuf_generate_cpp(JAX_PROTO_SRCS JAX_PROTO_HDRS
  "${JAX_PROTO_DIR}/xplane.proto"
  "${JAX_PROTO_DIR}/perfetto_trace.proto")
