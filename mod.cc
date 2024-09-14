#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "lapacke.h"

void say_hello() { std::cout << "Hello, World!" << std::endl; }

void lapacke_solve() {
  int n = 3;
  double a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  double b[3] = {1, 2, 3};
  int ipiv[3];
  int info;

  info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, *a, n, ipiv, b, 1);

  if (info > 0) {
    std::cout << "The diagonal element of the triangular factor of A,\n";
    std::cout << "U(" << info << "," << info
              << ") is zero, so that A is singular;\n";
    std::cout << "the solution could not be computed.\n";
    return;
  }

  std::cout << "Solution: " << b[0] << ", " << b[1] << ", " << b[2]
            << std::endl;
}

void lapacke_solve1(const std::vector<std::vector<double>> &A,
                    const std::vector<double> &B) {
  int n = B.size();

  double a[n][n];
  // copy A to a
  // 将 vector 中的值拷贝到 mat 中
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i][j] = A[i][j];
    }
  }
  double b[n];
  // copy B to b
  std::copy(B.begin(), B.end(), b);

  int ipiv[n];
  int info;

  info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, *a, n, ipiv, b, 1);

  if (info > 0) {
    std::cout << "The diagonal element of the triangular factor of A,\n";
    std::cout << "U(" << info << "," << info
              << ") is zero, so that A is singular;\n";
    std::cout << "the solution could not be computed.\n";
    return;
  }
  std::cout << "Solution: " << b[0] << ", " << b[1] << ", " << b[2]
            << std::endl;
}

PYBIND11_MODULE(pycppmod, m) {
  m.def("say_hello", &say_hello);
  m.def("lapacke_solve", &lapacke_solve);
  m.def("lapacke_solve1", &lapacke_solve1);
}