#include "Eigen/Dense"
#include <algorithm>
#include <iostream>
#include <pybind11/eigen.h>
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

void lapacke_lu() {

  int n = 10;
  int lda = n;
  double a[n * n];
  std::fill(a, a + n * n, 0);

  std::vector<double> values = {
      1089.942113770775,  1010.096696063498, 58.66178651250262,
      5661.020957489456,  -1589555826.24471, -3799865302.243693,
      -162987547.4735685, -1771081.02947091, 255.3627654232152,
      151.2319507163488,
  };

  for (int i = 0; i < n; ++i) {
    a[i * n + i] = values[i];
  }

  for (int i = 0; i < n; ++i) {
    std::cout << "[ ";
    for (int j = 0; j < n; ++j) {
      std::cout << a[i * n + j] << " ";
    }
    std::cout << " ]" << std::endl;
  }

  int info;

  int ipiv[n];
  info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ipiv);

  if (info > 0) {
    std::cout << "The diagonal element of the triangular factor of A,\n";
    std::cout << "U(" << info << "," << info
              << ") is zero, so that A is singular;\n";
    std::cout << "the solution could not be computed.\n";
    return;
  }

  std::cout << "LU factorization: " << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << a[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

void lapacke_lu1(Eigen::VectorXd values) {

  int n = 10;
  int lda = n;
  double a[n * n];
  std::fill(a, a + n * n, 0);

  for (int i = 0; i < n; ++i) {
    a[i * n + i] = values[i];
  }

  for (int i = 0; i < n; ++i) {
    std::cout << "[ ";
    for (int j = 0; j < n; ++j) {
      std::cout << a[i * n + j] << " ";
    }
    std::cout << " ]" << std::endl;
  }

  int info;

  int ipiv[n];
  info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ipiv);

  if (info > 0) {
    std::cout << "The diagonal element of the triangular factor of A,\n";
    std::cout << "U(" << info << "," << info
              << ") is zero, so that A is singular;\n";
    std::cout << "the solution could not be computed.\n";
    return;
  }

  std::cout << "LU factorization: " << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << a[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

void lapacke_lu2(Eigen::MatrixXd &values) {

  int n = values.rows();
  int lda = n;
  double a[n * n];
  std::fill(a, a + n * n, 0);

  std::copy(values.data(), values.data() + n * n, a);

  // for (int i = 0; i < n; ++i) {
  //   a[i * n + i] = values(i, i);
  // }

  for (int i = 0; i < n; ++i) {
    std::cout << "[ ";
    for (int j = 0; j < n; ++j) {
      std::cout << a[i * n + j] << " ";
    }
    std::cout << " ]" << std::endl;
  }

  int info;

  int ipiv[n];
  info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ipiv);

  if (info > 0) {
    std::cout << "The diagonal element of the triangular factor of A,\n";
    std::cout << "U(" << info << "," << info
              << ") is zero, so that A is singular;\n";
    std::cout << "the solution could not be computed.\n";
    return;
  }

  std::cout << "LU factorization: " << std::endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << a[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

PYBIND11_MODULE(pycppmod, m) {
  m.def("say_hello", &say_hello);
  m.def("lapacke_solve", &lapacke_solve);
  m.def("lapacke_solve1", &lapacke_solve1);
  m.def("lu", &lapacke_lu);
  m.def("lu1", &lapacke_lu1);
  m.def("lu2", &lapacke_lu2);
}
