/*
    一些计算过程用到的核函数，可同时用于单gpu和多gpu计算
*/

#ifndef KERNELS_H
#define KERNELS_H

#include <device_launch_parameters.h>
#include "launch.h"

/*
    计算能力6.0以下的设备无双精度浮点类型的原子操作，以下为atomicAdd的实现。
    注意：atomicCAS为atomic compare and swap，不是atomic compare and set
         atomicCAS(address, expected, val)计算*address == expected ? val : *address，
         并把结果存储到address，具体效果为：
         1、如果*address等于expected，则把val存入address，返回expected
         2、如果*address不等于expected，则直接返回*address
*/


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double *address, double val) {
    unsigned long long *address_as_ull = (unsigned long long *) address;
    unsigned long long old = *address_as_ull;
    unsigned long long expected;

    do {
        expected = old;
        old = atomicCAS(address_as_ull, expected,
                        __double_as_longlong(val + __longlong_as_double(expected)));
    } while (expected != old);

    return __longlong_as_double(old);
}
#endif


/*
    以相同的值填充稠密向量的核函数
*/
template<typename ValueType, typename IndexType>
__global__ void FillKernel(ValueType *vec, const IndexType N, const ValueType value) {
    const int stride = blockDim.x * gridDim.x;
    for (IndexType i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
        vec[i] = value;
}


/*
    将csr存储格式的row_offsets转换为coo储存格式的row_indices
*/
template<typename IndexType>
__global__ void Csr2CooKernel(IndexType *row_indices,
                              const IndexType *row_offsets,
                              const IndexType num_rows) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane = thread_id & 15;
    const int num_rows_per_block = (blockDim.x * gridDim.x) / 16;

    for (IndexType row = thread_id / 16; row < num_rows; row += num_rows_per_block) {
        for (IndexType j = row_offsets[row] + lane; j < row_offsets[row + 1]; j += 16)
            row_indices[j] = row;
    }
}


/*
    将coo存储格式的row_indices转换为coo储存格式的row_offsets
    注意：这个kernel只能完成统计各行非零元素个数的工作，完整的Coo2Csr还需要一次scan
*/
template<typename IndexType>
__global__ void Coo2CsrKernel1(IndexType *row_offsets,
                               const IndexType *row_indices,
                               const IndexType num_entries) {
    const int stride = blockDim.x * gridDim.x;
    for (IndexType i = threadIdx.x + blockIdx.x * blockDim.x; i < num_entries; i += stride) {
        atomicAdd(&row_offsets[row_indices[i] + 1], 1);
    }
}


/*
    统计一个csr格式的稀疏矩阵哪些列有非零元素，结果保存在长度为num_cols的数组columns_count中，值为1的是
    有非零元素的列，值为0的是无非零元素的列，注意需要先把columns_count的初始值置为0
*/
template<typename IndexType>
__global__ void ColumnsCountKernel(int *columns_count,
                                   const IndexType *column_indices,
                                   const IndexType num_entries) {
    const int stride = blockDim.x * gridDim.x;
    for (IndexType i = threadIdx.x + blockDim.x * blockIdx.x; i < num_entries; i += stride)
        columns_count[column_indices[i]] = 1;
}


/*
    归约内核，用于对一个数组进行求和
*/
template<typename ValueType, typename IndexType, int BlockSize = BLOCK_SIZE>
__global__ void ReductionKernel(ValueType *partial,
                                const ValueType *vec,
                                const IndexType N) {
    __shared__ volatile ValueType cache[BlockSize];

    int tid = threadIdx.x;

    ValueType t = 0;
    for (IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
                i < N; i += gridDim.x * blockDim.x)
        t += vec[i];
    cache[tid] = t;
    __syncthreads();

    // 线程块内归约
    if (BlockSize >= 1024 && tid < 512)
        cache[tid] += cache[tid + 512];
    __syncthreads();

    if (BlockSize >= 512 && tid < 256)
        cache[tid] += cache[tid + 256];
    __syncthreads();

    if (BlockSize >= 256 && tid < 128)
        cache[tid] += cache[tid + 128];
    __syncthreads();

    if (BlockSize >= 128 && tid < 64)
        cache[tid] += cache[tid + 64];
    __syncthreads();

    // 线程束内归约，指令本身是同步的，不需要调用__syncthreads，每步只保证前半部分数据正确，
    // 后半部分数据错误但已经不需要了
    if (tid < 32) {
        cache[tid] += cache[tid + 32];
        cache[tid] += cache[tid + 16];
        cache[tid] += cache[tid + 8];
        cache[tid] += cache[tid + 4];
        cache[tid] += cache[tid + 2];
        cache[tid] += cache[tid + 1];
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}


/*
    计算两个向量内积的核函数，线程块内归约，将部分和放在partial中，所以partial的长度至少
    应该为线程块数。
    假设块内线程数量是2的指数且至少为64。
*/
template<typename ValueType, typename IndexType, int BlockSize = BLOCK_SIZE>
__global__ void DotKernel(ValueType *partial,
                          const ValueType *vec1,
                          const ValueType *vec2,
                          const IndexType N) {
    __shared__ volatile ValueType cache[BlockSize];

    int tid = threadIdx.x;

    ValueType t = 0;
    for (IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
                i < N; i += gridDim.x * blockDim.x)
        t += vec1[i] * vec2[i];
    cache[tid] = t;
    __syncthreads();

    // 线程块内归约
    if (BlockSize >= 1024 && tid < 512)
        cache[tid] += cache[tid + 512];
    __syncthreads();

    if (BlockSize >= 512 && tid < 256)
        cache[tid] += cache[tid + 256];
    __syncthreads();

    if (BlockSize >= 256 && tid < 128)
        cache[tid] += cache[tid + 128];
    __syncthreads();

    if (BlockSize >= 128 && tid < 64)
        cache[tid] += cache[tid + 64];
    __syncthreads();

    // 线程束内归约，指令本身是同步的，不需要调用__syncthreads，每步只保证前半部分数据正确，
    // 后半部分数据错误但已经不需要了
    if (tid < 32) {
        cache[tid] += cache[tid + 32];
        cache[tid] += cache[tid + 16];
        cache[tid] += cache[tid + 8];
        cache[tid] += cache[tid + 4];
        cache[tid] += cache[tid + 2];
        cache[tid] += cache[tid + 1];
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}


/*
    计算csr格式的稀疏矩阵与稠密向量之积的核函数: result = Av
    1/4个warp计算一个分量
*/
template<typename ValueType, typename IndexType, int BlockSize = BLOCK_SIZE>
__global__ void SpMvKernel(ValueType *result,
                           const ValueType *values,
                           const IndexType *row_offsets,
                           const IndexType *column_indices,
                           const IndexType num_rows,
                           const ValueType *vec) {
    __shared__ volatile ValueType cache[BlockSize];

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 7;           // 1/4线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 8;

    for (IndexType row = thread_id / 8; row < num_rows; row += num_rows_per_block) {
        ValueType t = 0;
        for (IndexType j = row_offsets[row] + lane; j < row_offsets[row + 1]; j += 8)
            t += values[j] * vec[column_indices[j]];
        cache[threadIdx.x] = t;

        // parallel reduction in shared memory
        if (lane < 4) {
            // cache[threadIdx.x] = t = t + cache[threadIdx.x + 8];
            cache[threadIdx.x] = t = t + cache[threadIdx.x + 4];
            cache[threadIdx.x] = t = t + cache[threadIdx.x + 2];
            t = t + cache[threadIdx.x + 1];
        }

        // first thread writes the result
        if (lane == 0)
            result[row] = t;
    }
}


/*
    计算csr格式的稀疏矩阵的转置与稠密向量的乘积的核函数：result = A^T v
    将此乘法看作以v的每一个分量为
    半个线程束处理A的一行
*/
template<typename ValueType, typename IndexType>
__global__ void SpMTvKernel(ValueType *result,
                            const ValueType *values,
                            const IndexType *row_offsets,
                            const IndexType *column_indices,
                            const IndexType num_rows,
                            const ValueType *vec) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 15;           // 线程束内偏移量
    const int num_warps = gridDim.x * blockDim.x / 16;

    for (int row = thread_id / 16; row < num_rows; row += num_warps) {
        for (int j = row_offsets[row] + lane; j < row_offsets[row + 1]; j += 16)
            atomicAdd(&result[column_indices[j]], vec[row] * values[j]);
    }
}


/*
    计算两个稠密向量的距离的核函数，线程块内归约，因此partial必须至少要与grid_size一样大
*/
template<typename ValueType, typename IndexType, int BlockSize = BLOCK_SIZE>
__global__ void DistanceKernel(ValueType *partial,
                               const ValueType *vec1,
                               const ValueType *vec2,
                               const IndexType N) {
    __shared__ volatile ValueType cache[BlockSize];

    int tid = threadIdx.x;

    ValueType t = 0;
    for (IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
                i < N; i += gridDim.x * blockDim.x)
        t += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    cache[tid] = t;
    __syncthreads();

    // 线程块内归约
    if (BlockSize >= 1024 && tid < 512)
        cache[tid] += cache[tid + 512];
    __syncthreads();

    if (BlockSize >= 512 && tid < 256)
        cache[tid] += cache[tid + 256];
    __syncthreads();

    if (BlockSize >= 256 && tid < 128)
        cache[tid] += cache[tid + 128];
    __syncthreads();

    if (BlockSize >= 128 && tid < 64)
        cache[tid] += cache[tid + 64];
    __syncthreads();

    // 线程束内归约，指令本身是同步的，不需要调用__syncthreads，每步只保证前半部分数据正确，
    // 后半部分数据错误但已经不需要了
    if (tid < 32) {
        cache[tid] += cache[tid + 32];
        cache[tid] += cache[tid + 16];
        cache[tid] += cache[tid + 8];
        cache[tid] += cache[tid + 4];
        cache[tid] += cache[tid + 2];
        cache[tid] += cache[tid + 1];
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}


/*
    计算稠密向量数乘的核函数: result = alpha * vec
*/
template<typename ValueType, typename IndexType>
__global__ void ScaleKernel(ValueType *result,
                            const ValueType *vec,
                            const IndexType N,
                            const ValueType alpha) {
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
        result[i] = alpha * vec[i];
}


/*
    计算稠密向量加法的核函数: result = v1 + alpha * v2
*/
template<typename ValueType, typename IndexType>
__global__ void AddKernel1(ValueType *result,
                           const ValueType *v1,
                           const ValueType *v2,
                           const IndexType N,
                           const ValueType alpha) {
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
        result[i] = v1[i] + alpha * v2[i];
}


/*
    计算稠密向量加法的核函数: result = alpha * v1 + beta * v2
*/
template<typename ValueType, typename IndexType>
__global__ void AddKernel2(ValueType *result,
                           const ValueType *v1,
                           const ValueType *v2,
                           const IndexType N,
                           const ValueType alpha,
                           const ValueType beta) {
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
        result[i] = alpha * v1[i] + beta * v2[i];
}


/*
    稠密向量加上一个常数: result[i] = v[i] + c
*/
template<typename ValueType, typename IndexType>
__global__ void AddKernel3(ValueType *result,
                           const ValueType *v,
                           const ValueType c,
                           const IndexType N) {
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
        result[i] = v[i] + c;
}


/*
    计算稠密向量加法的核函数: result[i] = alpha * v1[i] + beta * v2[i] * v3[i]
*/
template<typename ValueType, typename IndexType>
__global__ void AddKernel4(ValueType *result,
                           const ValueType *v1,
                           const ValueType *v2,
                           const ValueType *v3,
                           const IndexType N,
                           const ValueType alpha,
                           const ValueType beta) {
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
        result[i] = alpha * v1[i] + beta * v2[i] * v3[i];
}


/*
    计算两个向量的Hadamard积，即各对应元素相乘
*/
template<typename ValueType, typename IndexType>
__global__ void HadamardKernel(ValueType *result,
                               const ValueType *v1,
                               const ValueType *v2,
                               const IndexType N) {
    const int stride = blockDim.x * gridDim.x;
    for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        result[i] = v1[i] * v2[i];
}


/*
    计算残量的核函数：result = b - Ax
    1/4个warp计算一个分量
*/
template<typename ValueType, typename IndexType, int BlockSize = BLOCK_SIZE>
__global__ void ResidualKernel(ValueType *r,
                               const ValueType *values,
                               const IndexType *row_offsets,
                               const IndexType *column_indices,
                               const IndexType num_rows,
                               const ValueType *b,
                               const ValueType *x) {
    __shared__ volatile ValueType cache[BlockSize];

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 7;           // 1/4线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 8;

    for (IndexType row = thread_id / 8; row < num_rows; row += num_rows_per_block) {
        ValueType t = 0;
        for (IndexType j = row_offsets[row] + lane; j < row_offsets[row + 1]; j += 8)
            t += values[j] * x[column_indices[j]];
        cache[threadIdx.x] = t;

        // parallel reduction in shared memory
        if (lane < 4) {
            // cache[threadIdx.x] = t = t + cache[threadIdx.x + 8];
            cache[threadIdx.x] = t = t + cache[threadIdx.x + 4];
            cache[threadIdx.x] = t = t + cache[threadIdx.x + 2];
            t = t + cache[threadIdx.x + 1];
        }

        // first thread writes the result
        if (lane == 0)
            r[row] = b[row] - t;
    }
}


/*
    计算稀疏矩阵对角元素，半个线程束计算一个
*/
template<typename ValueType, typename IndexType>
__global__ void FindDiag(ValueType *diag,
                         const ValueType *values,
                         const IndexType *row_offsets,
                         const IndexType *column_indices,
                         const IndexType num_rows) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 15;           // 半线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 16;

    for (IndexType row = thread_id / 16; row < num_rows; row += num_rows_per_block) {
        for (IndexType j = row_offsets[row] + lane;
                    j < row_offsets[row + 1] && column_indices[j] <= row; j += 16) {
            if (column_indices[j] == row)
                diag[row] = values[j];
        }
    }
}

/*
    计算元素绝对值的最小值，配合稀疏矩阵对角元素计算的核函数可以得到对角元素绝对值的最小值
*/
template<typename ValueType, typename IndexType, int BlockSize = BLOCK_SIZE>
__global__ void MinAbsKernel(ValueType *partial,
                             const ValueType *vec,
                             const IndexType N) {
    __shared__ volatile ValueType cache[BlockSize];

    int tid = threadIdx.x;

    ValueType t = INFINITY;
    for (IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
                i < N; i += gridDim.x * blockDim.x)
        t = min(t, abs(vec[i]));
    cache[tid] = t;
    __syncthreads();

    // 线程块内归约
    if (BlockSize >= 1024 && tid < 512)
        cache[tid] = min(cache[tid], cache[tid + 512]);
    __syncthreads();

    if (BlockSize >= 512 && tid < 256)
        cache[tid] = min(cache[tid], cache[tid + 256]);
    __syncthreads();

    if (BlockSize >= 256 && tid < 128)
        cache[tid] = min(cache[tid], cache[tid + 128]);
    __syncthreads();

    if (BlockSize >= 128 && tid < 64)
        cache[tid] = min(cache[tid], cache[tid + 64]);
    __syncthreads();

    // 线程束内归约，指令本身是同步的，不需要调用__syncthreads，每步只保证前半部分数据正确，
    // 后半部分数据错误但已经不需要了
    if (tid < 32) {
        cache[tid] = min(cache[tid], cache[tid + 32]);
        cache[tid] = min(cache[tid], cache[tid + 16]);
        cache[tid] = min(cache[tid], cache[tid + 8]);
        cache[tid] = min(cache[tid], cache[tid + 4]);
        cache[tid] = min(cache[tid], cache[tid + 2]);
        cache[tid] = min(cache[tid], cache[tid + 1]);
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}


/*
    Jacobi迭代的核函数，16个线程算一行
*/
template<typename ValueType, typename IndexType, int BlockSize = BLOCK_SIZE>
__global__ void JacobiKernel(ValueType *x2,
                             const ValueType *x1,
                             const ValueType *values,
                             const IndexType *row_offsets,
                             const IndexType *column_indices,
                             const IndexType num_rows,
                             const ValueType *b) {
    __shared__ volatile ValueType cache[BlockSize];
    __shared__ volatile ValueType Aii[BlockSize / 16];   // 保存对角线元素

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 15;                     // 半线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 16;

    for (IndexType row = thread_id / 16; row < num_rows; row += num_rows_per_block) {
        ValueType t = 0;
        for (IndexType j = row_offsets[row] + lane; j < row_offsets[row + 1]; j += 16) {
            if (row != column_indices[j])
                t += values[j] * x1[column_indices[j]];
            else
                Aii[threadIdx.x / 16] = values[j];
        }
        cache[threadIdx.x] = t;

        // parallel reduction in shared memory
        if (lane < 8) {
            cache[threadIdx.x] = t = t + cache[threadIdx.x + 8];
            cache[threadIdx.x] = t = t + cache[threadIdx.x + 4];
            cache[threadIdx.x] = t = t + cache[threadIdx.x + 2];
            t = t + cache[threadIdx.x + 1];
        }

        // first thread writes the result
        if (lane == 0)
            x2[row] = (b[row] - t) / Aii[threadIdx.x / 16];
    }
}


/*
    计算稀疏矩阵对角元素的逆
*/
template<typename ValueType, typename IndexType>
__global__ void FindDiagInv(ValueType *diag_inv,
                            const ValueType *values,
                            const IndexType *row_offsets,
                            const IndexType *column_indices,
                            const IndexType num_rows) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 15;           // 半线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 16;

    for (IndexType row = thread_id / 16; row < num_rows; row += num_rows_per_block) {
        for (IndexType j = row_offsets[row] + lane;
                    j < row_offsets[row + 1] && column_indices[j] <= row; j += 16) {
            if (column_indices[j] == row)
                diag_inv[row] = 1.0 / values[j];
        }
    }
}


/*
    计算稀疏矩阵对角线部分的平方根的逆
*/
template<typename ValueType, typename IndexType>
__global__ void FindDiagInvSqrt(ValueType *diag_inv_sqrt,
                                const ValueType *values,
                                const IndexType *row_offsets,
                                const IndexType *column_indices,
                                const IndexType num_rows) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 15;           // 半线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 16;

    for (IndexType row = thread_id / 16; row < num_rows; row += num_rows_per_block) {
        for (IndexType j = row_offsets[row] + lane;
                    j < row_offsets[row + 1] && column_indices[j] <= row; j += 16) {
            if (column_indices[j] == row)
                diag_inv_sqrt[row] = sqrt(1.0 / values[j]);
        }
    }
}


/*
    计算稀疏矩阵对角线部分的平方根的逆
*/
template<typename ValueType, typename IndexType>
__global__ void FindDiagSqrt(ValueType *diag_sqrt,
                             const ValueType *values,
                             const IndexType *row_offsets,
                             const IndexType *column_indices,
                             const IndexType num_rows) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 15;           // 半线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 16;

    for (IndexType row = thread_id / 16; row < num_rows; row += num_rows_per_block) {
        for (IndexType j = row_offsets[row] + lane;
                    j < row_offsets[row + 1] && column_indices[j] <= row; j += 16) {
            if (column_indices[j] == row)
                diag_sqrt[row] = sqrt(values[j]);
        }
    }
}


/*
    多GPU计算稀疏矩阵对角线部分的平方根的逆，遍历行时row+first_row才是它在矩阵中的真正行号
*/
template<typename ValueType, typename IndexType>
__global__ void FindDiagInvSqrtMultiGPU(ValueType *diag_inv_sqrt,
                                        const ValueType *values,
                                        const IndexType *row_offsets,
                                        const IndexType *column_indices,
                                        const IndexType num_rows,
                                        const IndexType first_row) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = thread_id & 15;           // 半线程束内偏移量
    const int num_rows_per_block = gridDim.x * blockDim.x / 16;

    for (IndexType row = thread_id / 16; row < num_rows; row += num_rows_per_block) {
        for (IndexType j = row_offsets[row] + lane;
                    j < row_offsets[row + 1] && column_indices[j] <= row; j += 16) {
            if (column_indices[j] == row + first_row)
                diag_inv_sqrt[row] = sqrt(1.0 / values[j]);
        }
    }
}


#endif // KERNELS_H
