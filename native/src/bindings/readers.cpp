#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bindings/bindings.hpp"
#include "bindings/convert.hpp"
#include "readers/chakra/reader.hpp"
#include "readers/goal/reader.hpp"
#include "readers/kineto/reader.hpp"
#include "readers/nsys/reader.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace cta::bindings {
namespace {

// Runs a reader without the GIL and returns its columns as a dict.
template <typename Read>
nb::dict read_without_gil(Read&& read) {
  decltype(read()) columns;
  {
    nb::gil_scoped_release release;
    columns = read();
  }
  return to_dict(std::move(columns));
}

}  // namespace

void bind_readers(nb::module_& m) {
  m.def(
      "read_kineto",
      [](const std::string& path, int threads, std::uint64_t chunk_bytes, std::uint64_t block_bytes) {
        return read_without_gil([&] { return readers::read_kineto(path, threads, chunk_bytes, block_bytes); });
      },
      "path"_a, "threads"_a = 0, "chunk_bytes"_a = 32ull << 20, "block_bytes"_a = 8ull << 20,
      "Reads a Kineto trace (.json or .json.gz) into dictionary-coded columns.");

  m.def(
      "read_chakra",
      [](const std::vector<std::string>& paths, const std::vector<std::uint32_t>& ranks, int threads) {
        return read_without_gil([&] { return readers::read_chakra(paths, ranks, threads); });
      },
      "paths"_a, "ranks"_a, "threads"_a = 0,
      "Reads Chakra .et files (one rank each) into dictionary-coded node and edge columns.");

  m.def(
      "read_goal",
      [](const std::string& path, int threads, std::uint64_t chunk_bytes) {
        return read_without_gil([&] { return readers::read_goal(path, threads, chunk_bytes); });
      },
      "path"_a, "threads"_a = 0, "chunk_bytes"_a = 32ull << 20,
      "Reads a GOAL text schedule (.goal) into node and edge columns.");

  m.def(
      "read_nsys",
      [](const std::string& path) { return read_without_gil([&] { return readers::read_nsys(path); }); },
      "path"_a,
      "Reads the GPU activities, NVTX ranges and GPU metrics of an Nsight Systems SQLite export.");
}

}  // namespace cta::bindings
