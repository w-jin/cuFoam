#ifndef CHECK_H
#define CHECK_H

#include <cstdio>
#include <cuda_runtime.h>

// 在某个点检查前面的代码是否出错
#define CHECK_ERROR()                                                          \
do {                                                                           \
    const cudaError_t error = cudaGetLastError();                              \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// 检查某个函数调用是否出错
#define CHECK(call)                                                            \
do {                                                                           \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define CHECK_CUBLAS(call)                                                     \
do {                                                                           \
    cublasStatus_t error = call;                                               \
    if (error != CUBLAS_STATUS_SUCCESS)                                        \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", error, __FILE__,     \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define CHECK_CURAND(call)                                                     \
do {                                                                           \
    curandStatus_t error = call;                                               \
    if (error != CURAND_STATUS_SUCCESS)                                        \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", error, __FILE__,     \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define CHECK_CUFFT(call)                                                      \
do {                                                                           \
    cufftResult error = call;                                                  \
    if (error != CUFFT_SUCCESS)                                                \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", error, __FILE__,      \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define CHECK_CUSPARSE(call)                                                   \
do {                                                                           \
    cusparseStatus_t error = call;                                             \
    if (error != CUSPARSE_STATUS_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", error, __FILE__, __LINE__); \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#endif // CHECK_H

