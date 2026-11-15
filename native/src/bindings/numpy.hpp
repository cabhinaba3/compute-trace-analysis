// Hands C++ vectors to NumPy without copying.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace cta::bindings {

// The array owns the vector; NumPy frees it when the array dies.
template <typename T>
nanobind::object to_numpy(std::vector<T>&& values) {
  namespace nb = nanobind;
  if (values.empty()) {
    nb::ndarray<nb::numpy, T, nb::ndim<1>> array(nullptr, {0});
    return nb::cast(array);
  }
  auto* owned = new std::vector<T>(std::move(values));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  nb::ndarray<nb::numpy, T, nb::ndim<1>> array(owned->data(), {owned->size()}, owner);
  return nb::cast(array);
}

}  // namespace cta::bindings
