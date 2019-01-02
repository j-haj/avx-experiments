#include <chrono>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include <random>
#include <type_traits>
#include <vector>
#include <x86intrin.h>

#include <boost/align.hpp>

template <typename T, std::size_t Alignment = 32>
using aligned_vector = std::vector<T,
				   boost::alignment::aligned_allocator<T, Alignment>>;

template <typename T>
aligned_vector<T> random_vector(std::size_t n) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_real_distribution<T> dist(-1, 1);
  aligned_vector<T> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

template <typename T>
double distance(const aligned_vector<T>& x, const aligned_vector<T>& y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    auto d = x[i] - y[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

template <typename T>
void saxpy(T, aligned_vector<T>&, const aligned_vector<T>&, const aligned_vector<T>&);

template <typename T>
void avx_saxpy(T s, aligned_vector<T>& a, const aligned_vector<T>& x,
	       const aligned_vector<T>& y) {
  T* ap = a.data();
  const T* xp = x.data();
  const T* yp = y.data();
  if constexpr (std::is_same<T, double>::value) {
    // -- DOUBLE --
#ifdef __AVX2__
    // AVX2 has 256 bit ops, which gives us 4 doubles
    const int n_doubles = 4;
    std::size_t rem = a.size() % 4;
    std::size_t n = (a.size() - rem) / 4;
    for (std::size_t i = 0; i < n; ++i) {
      __m256d sm = _mm256_set_pd(s, s, s, s);
      __m256d am = _mm256_load_pd(ap + n_doubles * i);
      am = _mm256_mul_pd(am, sm);
      __m256d xm = _mm256_load_pd(xp + n_doubles * i);
      am = _mm256_mul_pd(am, xm);
      __m256d ym = _mm256_load_pd(yp + n_doubles * i);
      am = _mm256_add_pd(am, ym);
      _mm256_store_pd(ap + n_doubles * i, am);
    }
#else
    // AVX has 128 bit ops, which gives use 2 doubles
    const int n_doubles = 2;
    std::size_t rem = a.size() % 2;
    std::size_t n = (a.size() - rem) / 2;
    for (std::size_t i = 0; i < n; ++i) {
      __m128d sm = _mm_set_pd(s, s);
      __m128d am = _mm_load_pd(ap + n_doubles * i);
      am = _mm_mul_pd(am, sm);
      __m128d xm = _mm_load_pd(xp + n_doubles * i);
      am = _mm_mul_pd(am, xm);
      __m128d ym = _mm_load_pd(yp + n_doubles * i);
      am = _mm_add_pd(am, ym);
      _mm_store_pd(ap + n_doubles * i, am);
    }
#endif
    // Compiler should be able to unrull this...
    for (std::size_t i = a.size() - rem; i < a.size(); ++i) {
      a[i] = a[i] * s * x[i] + y[i];
    }
  } else {
    // -- FLOAT	--				      
#ifdef __AVX2__
    // AVX2 has 256 bit ops, which gives us 8 floats
    const int n_floats = 8;
    std::size_t rem = a.size() % 8;
    std::size_t n = (a.size() - rem) / 8;
    for (std::size_t i = 0; i < n; ++i) {
      __m256 sm = _mm256_set_ps(s, s, s, s, s, s, s, s);
      __m256 am = _mm256_load_ps(ap + n_floats * i);
      am = _mm256_mul_ps(am, sm);
      __m256 xm = _mm256_load_ps(xp + n_floats * i);
      am = _mm256_mul_ps(am, xm);
      __m256 ym = _mm256_load_ps(yp + n_floats * i);
      am = _mm256_add_ps(am, ym);
      _mm256_store_ps(ap + n_floats * i, am);
    }
#else
    // AVX has 128 bit ops, which gives us 4 floats
    const int n_floats = 4;
    std::size_t rem = a.size() % 4;
    std::size_t n = (a.size() - rem) / 4;
    for (std::size_t i = 0; i < n; ++i) {
      __m128 sm = _mm_set_ps(s, s, s, s);
      __m128 am = _mm_load_ps(ap + n_floats * i);
      am = _mm_mul_ps(am, sm);
      __m128 xm = _mm_load_ps(xp + n_floats * i);
      am = _mm_mul_ps(am, xm);
      __m128 ym = _mm_load_ps(yp + n_floats * i);
      am = _mm_add_ps(am, ym);
      _mm_store_ps(ap + n_floats * i, am);
    }
#endif
    for (std::size_t i = a.size() - rem; i < a.size(); ++i) {
      a[i] = a[i] * s * x[i] + y[i];
    }
  }
}

template <typename T>
void saxpy(T s, aligned_vector<T>& a, const aligned_vector<T>& x,
	   const aligned_vector<T>& y) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = s * a[i] * x[i] + y[i];
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const aligned_vector<T>& v) {
  os << "{ ";
  for (const auto& x : v) {
    os << x << ' ';
  }
  os << "}";
  return os;
}

int main() {
#ifdef __AVX2__
  std::cout << "AVX2 available\n";
#else
  std::cout << "Using AVX\n";
#endif
  using t = double;
  const auto dim = 500;
  const auto n = 10000;
  const t s = 3.0;
  std::vector<aligned_vector<t>> a1(n);
  for (auto i = 0; i < n; ++i) {
    a1[i] = random_vector<t>(dim);
  }
  std::vector<aligned_vector<t>> a2(a1);

  std::vector<aligned_vector<t>> xs(n);
  std::vector<aligned_vector<t>> ys(n);
  for (auto i = 0; i < n; ++i) {
    xs[i] = random_vector<t>(dim);
    ys[i] = random_vector<t>(dim);
  }

  std::cout << "Starting AVX saxpy\n";
  auto start = std::chrono::steady_clock::now();
  for (auto i = 0; i < n; ++i) {
    avx_saxpy(s, a1[i], xs[i], ys[i]);
  }
  auto stop = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  std::cout << "AVX done in " << elapsed.count() << "s\n";
  
  start = std::chrono::steady_clock::now();
  for (auto i = 0; i < n; ++i) {
    saxpy(s, a2[i], xs[i], ys[i]);
  }
  stop = std::chrono::steady_clock::now();
  elapsed = stop - start;
  std::cout << "Standard done in " << elapsed.count() << "s\n";

  // Check difference
  double sum;
  for (auto i = 0; i < n; ++i) {
    sum += distance(a1[i], a2[i]);
  }
  std::cout << "Total difference: " << sum << '\n';
  
  return 0;
}

