// Hands C++ vectors to NumPy without copying.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace cta::bindings {

template <typename T>
struct NumpyDtype;
template <>
struct NumpyDtype<std::int8_t> {
  static constexpr const char* name = "int8";
};
template <>
struct NumpyDtype<std::int32_t> {
  static constexpr const char* name = "int32";
};
template <>
struct NumpyDtype<std::int64_t> {
  static constexpr const char* name = "int64";
};
template <>
struct NumpyDtype<double> {
  static constexpr const char* name = "float64";
};

// The array owns the vector; NumPy frees it when the array dies.
template <typename T>
nanobind::object to_numpy(std::vector<T>&& values) {
  namespace nb = nanobind;
  if (values.empty()) return nb::module_::import_("numpy").attr("empty")(0, NumpyDtype<T>::name);
  auto* owned = new std::vector<T>(std::move(values));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  nb::ndarray<nb::numpy, T, nb::ndim<1>> array(owned->data(), {owned->size()}, owner);
  return nb::cast(array);
}

}  // namespace cta::bindings
