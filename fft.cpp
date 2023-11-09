#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace chrono;

using Real = double;
using Complex = complex<Real>;
using Array = vector<Real>;
using CArray = vector<Complex>;

size_t BitReverse(size_t n, int nbits) {
  size_t x = 0;
  for (int i = 0; i != nbits; ++i) {
    x |= ((n >> i) & 0x1) << ((nbits - i) - 1);
  }
  return x;
}

void ShuffleCoeff(const Array &x, CArray &y) {
  auto bits = int(log2(x.size()));
  for (size_t i = 0; i != x.size(); ++i) {
    y[i] = Complex(x[BitReverse(i, bits)], 0.f);
  }
}

CArray FFT(const Array &x) {
  const auto N = x.size();
  CArray y(N);
  ShuffleCoeff(x, y);
  auto levels = int(log2(N));
  for (int l = 1; l != levels + 1; ++l) {
    auto span = 1 << l;
    auto s2 = span >> 1;
    auto wm = exp(Complex(0.f, -M_PI / Real(s2)));
    auto w = Complex(1.f, 0.f);
    for (int s = 0; s < s2; ++s) {
      for (int k = s; k < N; k += span) {
        Complex u = y[k];
        y[k] = u + w * y[k + s2];
        y[k + s2] = u - w * y[k + s2];
      }
      w *= wm;
    }
  }
  return y;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << argv[0] << " <array size (power of two)>" << endl;
    exit(EXIT_FAILURE);
  }
  const size_t SIZE = stoull(argv[1]);
  if (log2(double(SIZE)) != floor(log2(double(SIZE)))) {
    cerr << "Size must be a power of two" << endl;
    exit(EXIT_FAILURE);
  }
  Array x(SIZE);
  iota(begin(x), end(x), Real(1));
  auto t1 = high_resolution_clock::now();
  auto y = FFT(x);
  auto t2 = high_resolution_clock::now();
  if (SIZE <= 32) {
    copy(begin(y), end(y), ostream_iterator<Complex>(cout, " "));
    cout << endl;
  }
  cout << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

  ofstream os("/dev/null");
  os << y[1];
  return 0;
}
