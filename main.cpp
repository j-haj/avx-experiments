#include <chrono>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include <random>
#include <type_traits>
#include <vector>
#include <x86intrin.h>

template <typename T>
std::vector<T> random_vector(std::size_t n) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_real_distribution<T> dist(-1, 1);
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

template <typename T>
double distance(const std::vector<T>& x, const std::vector<T>& y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    auto d = x[i] - y[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

template <typename T>
void saxpy(T, std::vector<T>&, const std::vector<T>&, const std::vector<T>&);

template <typename T>
void avx_saxpy(T s, std::vector<T>& a, std::vector<T>& x,
	       std::vector<T>& y) {
  T* ap = a.data();
  T* xp = x.data();
  T* yp = y.data();
  if (std::is_same<T, double>::value) {
    // -- DOUBLE --
#ifdef __AVX2__
    // AVX2 has 256 bit ops, which gives us 4 doubles
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
    for (std::size_t i = a.size() - rem; i < a.size(); ++i) {
      a[i] = a[i] * s * x[i] + y[i];
    }
#endif
  } else if (std::is_same<T, float>::value) {
    // -- FLOAT	--				      
#ifdef __AVX2__
    // AVX2 has 256 bit ops, which gives us 8 floats
#else
    // AVX has 128 bit ops, which gives us 4 floats
    const int n_floats = 4;
    std::size_t rem = a.size() % 4;
    std::size_t n = (a.size() - rem) / 4;
    for (std::size_t i = 0; i < n; ++i) {

    }
#endif
  } else {
    saxpy(s, a, x, y);
  }
}

template <typename T>
void saxpy(T s, std::vector<T>& a, const std::vector<T>& x,
	   const std::vector<T>& y) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = s * a[i] * x[i] + y[i];
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "{ ";
  for (const auto& x : v) {
    os << x << ' ';
  }
  os << "}";
  return os;
}

int main() {
  using t = double;
  const auto dim = 500;
  const auto n = 10000;
  const t s = 3.0;
  std::vector<std::vector<t>> a1(n);
  for (auto i = 0; i < n; ++i) {
    a1[i] = random_vector<t>(dim);
  }
  std::vector<std::vector<t>> a2(a1);

  std::vector<std::vector<t>> xs(n);
  std::vector<std::vector<t>> ys(n);
  for (auto i = 0; i < n; ++i) {
    xs[i] = random_vector<t>(dim);
    ys[i] = random_vector<t>(dim);
  }

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

