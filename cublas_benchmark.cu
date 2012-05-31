/*
 * Copyright (C) 2012  NISHIMURA Ryohei
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define CUDA_SAFE_CALL(E) do {                                          \
    cudaError_t e = (E);                                                \
    if (e != cudaSuccess) {                                             \
      printf("line %d: CUDA error: %s\n", __LINE__, cudaGetErrorString(e)); \
      exit(-2);                                                         \
    }                                                                   \
  } while (false)

#define CUBLAS_SAFE_CALL(S) do {                                \
    cublasStatus_t s = (S);                                     \
    if (s != CUBLAS_STATUS_SUCCESS) {                           \
      printf("line %d: CUBLAS error: %d\n", __LINE__, s);       \
      exit(-3);                                                 \
    }                                                           \
  } while (false)

#define CHECKZERO(X) do {                               \
    if ((X) <= 0) {                                     \
      printf("error: " #X " = %d <= 0!\n", (X));        \
      exit(-1);                                         \
    }                                                   \
  } while(false)

#define CHECKMINMAX(X, Y) do {                                  \
    if ((X) > (Y)) {                                            \
      printf("error: " #X " = %d > %d = " #Y "!\n", (X), (Y));  \
      exit(-1);                                                 \
    }                                                           \
  } while(false)

template<typename T> void
gemm(cublasHandle_t handle, int m, int n, int k,
     const T *a, int lda, const T *b, int ldb,
     T *c, int ldc);

template<> void
gemm<float>(cublasHandle_t handle, int m, int n, int k,
            const float *a, int lda,
            const float *b, int ldb,
            float *c, int ldc)
{
  float alpha = 1.0f;
  float beta = 1.0f;
  CUBLAS_SAFE_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, k,
                               &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template<> void
gemm<double>(cublasHandle_t handle, int m, int n, int k,
             const double *a, int lda,
             const double *b, int ldb,
             double *c, int ldc)
{
  double alpha = 1.0;
  double beta = 1.0;
  CUBLAS_SAFE_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, k,
                               &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template<> void
gemm<cuComplex>(cublasHandle_t handle, int m, int n, int k,
                const cuComplex *a, int lda,
                const cuComplex *b, int ldb,
                cuComplex *c, int ldc)
{
  cuComplex alpha = make_cuComplex(1.0f, 0.0f);
  cuComplex beta = make_cuComplex(1.0f, 0.0f);
  CUBLAS_SAFE_CALL(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, k,
                               &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template<> void
gemm<cuDoubleComplex>(cublasHandle_t handle, int m, int n, int k,
                      const cuDoubleComplex *a, int lda,
                      const cuDoubleComplex *b, int ldb,
                      cuDoubleComplex *c, int ldc)
{
  cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
  CUBLAS_SAFE_CALL(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, k,
                               &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template<typename T> double
calc_gflops(int m, int n, int k, float ms);

template<> double
calc_gflops<float>(int m, int n, int k, float ms)
{
  return 2.0 * m * n * k / ms * 1.0e-6;
}

template<> double
calc_gflops<double>(int m, int n, int k, float ms)
{
  return 2.0 * m * n * k / ms * 1.0e-6;
}

template<> double
calc_gflops<cuComplex>(int m, int n, int k, float ms)
{
  return 8.0 * m * n * k / ms * 1.0e-6;
}

template<> double
calc_gflops<cuDoubleComplex>(int m, int n, int k, float ms)
{
  return 8.0 * m * n * k / ms * 1.0e-6;
}

template<typename T> void
benchmark(int m, int n, int k,
          int ldamin, int ldamax, int ldastep,
          int ldbmin, int ldbmax, int ldbstep,
          int ldcmin, int ldcmax, int ldcstep)
{
  cublasHandle_t handle;
  CUBLAS_SAFE_CALL(cublasCreate(&handle));
  T *a, *b, *c;
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&a),
                            k * ldamax * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemset(a, 0, k * ldamax * sizeof(T)));
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&b),
                            n * ldbmax * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemset(b, 0, n * ldbmax * sizeof(T)));
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&c),
                            n * ldcmax * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemset(c, 0, n * ldcmax * sizeof(T)));
  cudaStream_t stream;
  CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  CUBLAS_SAFE_CALL(cublasSetStream(handle, stream));
  cudaEvent_t start, end;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&end));
  printf("M: %d\tN: %d\tK: %d\n", m, n, k);
  printf("LDA\tLDB\tLDC\tSecond\tGFLOPS\n");
  for (int ldc = ldcmin; ldc <= ldcmax; ldc += ldcstep) {
    for (int ldb = ldbmin; ldb <= ldbmax; ldb += ldbstep) {
      for (int lda = ldamin; lda <= ldamax; lda += ldastep) {
        CUDA_SAFE_CALL(cudaEventRecord(start, stream));
        gemm(handle, m, n, k, a, lda, b, ldb, c, ldc);
        CUDA_SAFE_CALL(cudaEventRecord(end, stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(end));
        float ms;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, end));
        double gflops = calc_gflops<T>(m, n, k, ms);
        printf("%d\t%d\t%d\t%.6f\t%.6f\n", lda, ldb, ldc, ms, gflops);
      }
    }
  }
  CUDA_SAFE_CALL(cudaEventDestroy(end));
  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaStreamDestroy(stream));
  CUBLAS_SAFE_CALL(cublasDestroy(handle));
}

int
main(int argc, char **argv)
{
  if (argc < 14) {
    printf("usage: %s typename([sdcz]) m n k ldamin ldamax ldastep ldbmin ldbmax ldbstep ldcmin ldcmax ldcstep\n", argv[0]);
    return 0;
  }
  ptrdiff_t iarg = 0;
  char *type = argv[++iarg];
  int m = atoi(argv[++iarg]);
  int n = atoi(argv[++iarg]);
  int k = atoi(argv[++iarg]);
  int ldamin = atoi(argv[++iarg]);
  int ldamax = atoi(argv[++iarg]);
  int ldastep = atoi(argv[++iarg]);
  int ldbmin = atoi(argv[++iarg]);
  int ldbmax = atoi(argv[++iarg]);
  int ldbstep = atoi(argv[++iarg]);
  int ldcmin = atoi(argv[++iarg]);
  int ldcmax = atoi(argv[++iarg]);
  int ldcstep = atoi(argv[++iarg]);
  CHECKZERO(m);
  CHECKZERO(n);
  CHECKZERO(k);
  CHECKZERO(ldamin);
  CHECKMINMAX(m, ldamin);
  CHECKMINMAX(ldamin, ldamax);
  CHECKZERO(ldastep);
  CHECKZERO(ldbmin);
  CHECKMINMAX(k, ldbmin);
  CHECKMINMAX(ldbmin, ldbmax);
  CHECKZERO(ldbstep);
  CHECKZERO(ldcmin);
  CHECKMINMAX(m, ldcmin);
  CHECKMINMAX(ldcmin, ldcmax);
  CHECKZERO(ldcstep);

  switch (type[0]) {
  case 'S':
  case 's':
    benchmark<float>(m, n, k,
                     ldamin, ldamax, ldastep,
                     ldbmin, ldbmax, ldbstep,
                     ldcmin, ldcmax, ldcstep);
    break;
  case 'D':
  case 'd':
    benchmark<double>(m, n, k,
                      ldamin, ldamax, ldastep,
                      ldbmin, ldbmax, ldbstep,
                      ldcmin, ldcmax, ldcstep);
    break;
  case 'C':
  case 'c':
    benchmark<cuComplex>(m, n, k,
                         ldamin, ldamax, ldastep,
                         ldbmin, ldbmax, ldbstep,
                         ldcmin, ldcmax, ldcstep);
    break;
  case 'Z':
  case 'z':
    benchmark<cuDoubleComplex>(m, n, k,
                               ldamin, ldamax, ldastep,
                               ldbmin, ldbmax, ldbstep,
                               ldcmin, ldcmax, ldcstep);
    break;
  default:
    printf("invaild typename: %s\n", type);
    return -1;
  }

  return 0;
}
