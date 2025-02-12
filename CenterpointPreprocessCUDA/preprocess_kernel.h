#pragma once

#include "cuda_runtime_api.h"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#define CHECK_CUDA_ERROR(e) (check_error(e, __FILE__, __LINE__))

inline void check_error(const ::cudaError_t e, const char * f, int n)
{
  if (e != ::cudaSuccess) {
    ::std::stringstream s;
    s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": "
      << ::cudaGetErrorString(e);
    throw ::std::runtime_error{s.str()};
  }
}

struct deleter
{
  void operator()(void * p) const { CHECK_CUDA_ERROR(::cudaFree(p)); }
};

template <typename T>
using unique_ptr = ::std::unique_ptr<T, deleter>;

template <typename T>
typename ::std::enable_if<::std::is_array<T>::value, unique_ptr<T>>::type make_unique(
  const ::std::size_t n)
{
  using U = typename ::std::remove_extent<T>::type;
  U * p;
  CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(U) * n));
  return unique_ptr<T>{p};
}

namespace test{
std::size_t divup(const std::size_t a, const std::size_t b)  // cppcheck-suppress unusedFunction
{
  if (a == 0) {
    throw std::runtime_error("A dividend of divup isn't positive.");
  }
  if (b == 0) {
    throw std::runtime_error("A divisor of divup isn't positive.");
  }

  return (a + b - 1) / b;
}
}  // namespace test

// #endif  // AUTOWARE__LIDAR_CENTERPOINT__PREPROCESS__PREPROCESS_KERNEL_HPP_