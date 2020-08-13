#ifndef SOLVER_SINGLE_GPU_H
#define SOLVER_SINGLE_GPU_H

#include <algorithm>
#include <vector>
#include <utility>
#include <chrono>
#include <limits>
#include <cmath>
#include <cuda_runtime.h>

#include "SparseSingleGPU.h"
#include "check.h"
#include "kernels.h"
#include "launch.h"
#include "support.h"

/*
    单个GPU计算
*/
namespace SingleGPU {

/*
    一些不方便放在SparseSingleGPU.h中的操作
*/
// 以固定值v填充向量
template<typename ValueType, typename IndexType>
void Fill(Vector<ValueType, IndexType> &v, ValueType val) {
    LaunchKernel(FillKernel<ValueType, IndexType>, 0, 0, v.values, v.size, val);
}

// 转置
template<typename ValueType, typename IndexType>
SparseCSR<ValueType, IndexType> Transpose(const SparseCSR<ValueType, IndexType> &A) {
    // csr转为coo，只需要将row_offsets转为row_indices
    IndexType *row_indices = nullptr;
    CHECK(cudaMalloc(&row_indices, sizeof(IndexType) * A.num_entries));
    LaunchKernel(Csr2CooKernel<IndexType>, 0, 0, row_indices, A.row_offsets, A.num_rows);

    // 交换row_indices和column_indices
    SparseCSR<ValueType, IndexType> At(A.num_cols, A.num_rows, A.num_entries);
    CHECK(cudaMemcpy(At.values, A.values,
                     sizeof(ValueType) * A.num_entries, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(At.column_indices, row_indices,
                     sizeof(IndexType) * A.num_entries, cudaMemcpyDeviceToDevice));

    // 按照行排序，At的行即A的列
    IndexType *At_row_indices = nullptr;
    CHECK(cudaMalloc(&At_row_indices, sizeof(IndexType) * A.num_entries));
    CHECK(cudaMemcpy(At_row_indices, A.column_indices,
                     sizeof(IndexType) * A.num_entries, cudaMemcpyDeviceToDevice));
    thrust::stable_sort_by_key(thrust::device_ptr<IndexType>(At_row_indices),
                               thrust::device_ptr<IndexType>(At_row_indices + A.num_entries),
                               thrust::device_ptr<ValueType>(At.values));
    CHECK(cudaMemcpy(At_row_indices, A.column_indices,
                     sizeof(IndexType) * A.num_entries, cudaMemcpyDeviceToDevice));
    thrust::stable_sort_by_key(thrust::device_ptr<IndexType>(At_row_indices),
                               thrust::device_ptr<IndexType>(At_row_indices + A.num_entries),
                               thrust::device_ptr<IndexType>(At.column_indices));

    // 将At_row_indices转换成At.row_offsets，使用A.column_indices尽量避免争用
    CHECK(cudaMemset(At.row_offsets, 0, sizeof(IndexType) * (At.num_rows + 1)));
    LaunchKernel(Coo2CsrKernel1<IndexType>, 0, 0,
                 At.row_offsets, A.column_indices, At.num_entries);
    thrust::inclusive_scan(thrust::device_ptr<IndexType>(At.row_offsets),
                           thrust::device_ptr<IndexType>(At.row_offsets + At.num_rows + 1),
                           thrust::device_ptr<IndexType>(At.row_offsets));
    CHECK(cudaFree(row_indices));
    CHECK(cudaFree(At_row_indices));
    return At;
}

// 对数组进行求和
template<typename ValueType, typename IndexType>
ValueType Sum(const Vector<ValueType, IndexType> &v) {
    int grid_size = 0;
    int block_size = 0;
    LaunchConfig(&grid_size, &block_size);

    Vector<ValueType, IndexType> dev_partial;
    dev_partial.resize(grid_size);

    LaunchKernelWithGrid(ReductionKernel<ValueType, IndexType>,
                         grid_size, block_size, 0, 0,
                         dev_partial.data(), v.data(), v.size);

    CPU::Vector<ValueType, IndexType> partial;
    dev_partial.CopyToHost(partial);

    ValueType sum = 0;
    for (int i = 0; i < grid_size; i++)
        sum += partial[i];

    return sum;
}


/*
    return v1^T * v2
    由于归约需要在host端执行部分计算，所以不可能做成异步的，如果确实需要异步计算，
    可以考虑将其拆开，GPU部分可以异步执行
*/
template<typename ValueType, typename IndexType>
ValueType Dot(const Vector<ValueType, IndexType> &v1,
              const Vector<ValueType, IndexType> &v2) {
    if (v1.size != v2.size)
        throw std::runtime_error{"SingleGPU::Dot: 参与运算的向量维度不一致！"};

    int grid_size = 0;
    int block_size = 0;
    LaunchConfig(&grid_size, &block_size);

    Vector<ValueType, IndexType> dev_partial;
    dev_partial.resize(grid_size);

    LaunchKernelWithGrid(DotKernel<ValueType, IndexType>,
                         grid_size, block_size, 0, 0,
                         dev_partial.data(), v1.data(), v2.data(), v1.size);

    CPU::Vector<ValueType, IndexType> partial;
    dev_partial.CopyToHost(partial);

    ValueType sum = 0;
    for (int i = 0; i < grid_size; i++)
        sum += partial[i];

    return sum;
}

template<typename ValueType, typename IndexType>
ValueType DotThreadUnsafe(const Vector<ValueType, IndexType> &v1,
                          const Vector<ValueType, IndexType> &v2) {
    if (v1.size != v2.size)
        throw std::runtime_error{"SingleGPU::Dot: 参与运算的向量维度不一致！"};

    int grid_size = 0;
    int block_size = 0;
    LaunchConfig(&grid_size, &block_size);

    // ValueType *dev_partial = nullptr;
    // CHECK(cudaMalloc(&dev_partial, sizeof(ValueType) * grid_size));

    // 非线程安全
    static Vector<ValueType, IndexType> dev_partial;
    dev_partial.resize(grid_size);

    LaunchKernelWithGrid(DotKernel<ValueType, IndexType>,
                         grid_size, block_size, 0, 0,
                         dev_partial.data(), v1.data(), v2.data(), v1.size);

    // ValueType *partial = new ValueType[grid_size];
    // CHECK(cudaMemcpy(partial, dev_partial, sizeof(ValueType) * grid_size,
    //                     cudaMemcpyDeviceToHost));

    // 非线程安全
    static CPU::Vector<ValueType, IndexType> partial;
    dev_partial.CopyToHost(partial);

    ValueType sum = 0;
    for (int i = 0; i < grid_size; i++)
        sum += partial[i];

    // delete[] partial;
    // CHECK(cudaFree(dev_partial));

    return sum;
}


/*
    result = Av
*/
template<typename ValueType, typename IndexType>
void SpMv(Vector<ValueType, IndexType> *result,
          const SparseCSR<ValueType, IndexType> &A,
          const Vector<ValueType, IndexType> &v) {
    if (A.num_cols != v.size)
        throw std::runtime_error{"SingleGPU::SpMv: 参与运算的矩阵列数与向量维度不一致！"};

    result->resize(A.num_rows);
    LaunchKernel(SpMvKernel<ValueType, IndexType>, 0, 0,
                 result->data(),
                 A.values, A.row_offsets, A.column_indices, A.num_rows, v.data());
}


/*
    result = A^T * v
*/
template<typename ValueType, typename IndexType>
void SpMTv(Vector<ValueType, IndexType> *result,
           const SparseCSR<ValueType, IndexType> &A,
           const Vector<ValueType, IndexType> &v) {
    if (A.num_rows != v.size)
        throw std::runtime_error{"SingleGPU::SpMTv: 参与运算的矩阵行数与向量维度不一致！"};

    result->resize(A.num_cols);
    Fill(*result, 0.0);
    LaunchKernel(SpMTvKernel<ValueType, IndexType>, 0, 0,
                 result->data(),
                 A.values, A.row_offsets, A.column_indices, A.num_rows, v.data());
}


/*
    return sqrt(v^T * v), todo:to avoid probably overflow
*/
template<typename ValueType, typename IndexType>
ValueType Norm2(const Vector<ValueType, IndexType> &v) {
    return sqrt(Dot(v, v));
}


/*
    return (v1 - v2)^T * (v1 - v2)
*/
template<typename ValueType, typename IndexType>
ValueType Distance(const Vector<ValueType, IndexType> &v1,
                   const Vector<ValueType, IndexType> &v2) {
    if (v1.size != v2.size)
        throw std::runtime_error{"SingleGPU::Distance: 参与运算的向量维度不一致！"};

    int grid_size = 0;
    int block_size = 0;
    LaunchConfig(&grid_size, &block_size);

    Vector<ValueType, IndexType> dev_partial;
    dev_partial.resize(grid_size);

    LaunchKernelWithGrid(DistanceKernel<ValueType, IndexType>,
                         grid_size, block_size, 0, 0,
                         dev_partial.data(), v1.data(), v2.data(), v1.size);

    CPU::Vector<ValueType, IndexType> partial;
    dev_partial.CopyToHost(partial);

    ValueType sum = 0;
    for (int i = 0; i < grid_size; i++)
        sum += partial[i];

    return sqrt(sum);
}


/*
    result = alpha * v
*/
template<typename ValueType, typename IndexType>
void Scale(Vector<ValueType, IndexType> *result,
           const Vector<ValueType, IndexType> &v,
           const ValueType alpha) {
    result->resize(v.size);
    LaunchKernel(ScaleKernel<ValueType, IndexType>, 0, 0,
                 result->data(), v.data(), v.size, alpha);
}


/*
    result = v1 + alpha * v2
*/
template<typename ValueType, typename IndexType>
void Add(Vector<ValueType, IndexType> *result,
         const Vector<ValueType, IndexType> &v1,
         const Vector<ValueType, IndexType> &v2,
         const ValueType alpha = 1.0) {
    if (v1.size != v2.size)
        throw std::runtime_error{"SingleGPU::Add: 参与运算的两个向量长度不一致！"};
    result->resize(v1.size);
    LaunchKernel(AddKernel1<ValueType, IndexType>, 0, 0,
                 result->data(), v1.data(), v2.data(), v1.size, alpha);
}


/*
    result = alpha * v1 + beta * v2
*/
template<typename ValueType, typename IndexType>
void Add(Vector<ValueType, IndexType> *result,
         const Vector<ValueType, IndexType> &v1,
         const Vector<ValueType, IndexType> &v2,
         const ValueType alpha,
         const ValueType beta) {
    if (v1.size != v2.size)
        throw std::runtime_error{"SingleGPU::Add: 参与运算的两个向量长度不一致！"};
    result->resize(v1.size);
    LaunchKernel(AddKernel2<ValueType, IndexType>, 0, 0,
                 result->data(), v1.data(), v2.data(), v1.size, alpha, beta);
}


/*
    result[i] = alpha * v1[i] + beta * v2[i] * v3[i]
*/
template<typename ValueType, typename IndexType>
void Add(Vector<ValueType, IndexType> *result,
         const Vector<ValueType, IndexType> &v1,
         const Vector<ValueType, IndexType> &v2,
         const Vector<ValueType, IndexType> &v3,
         const ValueType alpha,
         const ValueType beta) {
    if (v1.size != v2.size || v1.size != v3.size)
        throw std::runtime_error{"SingleGPU::ADD: 参与运算的两个向量长度不一致！"};
    result->resize(v1.size);
    LaunchKernel(AddKernel4<ValueType, IndexType>, 0, 0,
                 result->data(), v1.data(), v2.data(), v3.data(), v1.size, alpha, beta);
}


/*
    计算两个向量的Hadamard积，即各对应元素相乘
    result[i] = v1[i] * v2[i]
*/
template<typename ValueType, typename IndexType>
void Hadamard(Vector<ValueType, IndexType> *result,
              const Vector<ValueType, IndexType> &v1,
              const Vector<ValueType, IndexType> &v2) {
    if (v1.size != v2.size)
        throw std::runtime_error{"SingleGPU::Add: 参与运算的两个向量长度不一致！"};
    result->resize(v1.size);
    LaunchKernel(HadamardKernel<ValueType, IndexType>, 0, 0,
                 result->data(), v1.data(), v2.data(), v1.size);
}


/*
    计算残量r = b - Ax，x和r所指内存区域有重叠时行为未定义
*/
template<typename ValueType, typename IndexType>
void Residual(Vector<ValueType, IndexType> *r,
              const SparseCSR<ValueType, IndexType> &A,
              const Vector<ValueType, IndexType> &b,
              const Vector<ValueType, IndexType> &x) {
    if (A.num_cols != x.size)
        throw std::runtime_error{"SingleGPU::Residual: A的列数与x维度不一致！"};
    if (A.num_rows != b.size)
        throw std::runtime_error{"SingleGPU::Residual: A的行数和b的维度不一致！"};

    r->resize(b.size);
    LaunchKernel(ResidualKernel<ValueType, IndexType>, 0, 0,
                 r->data(), A.values, A.row_offsets, A.column_indices, A.num_rows,
                 b.data(), x.data());
}


/*
    计算残量的二范数
*/
template<typename ValueType, typename IndexType>
ValueType ResidualNorm2(const SparseCSR<ValueType, IndexType> &A,
                        const Vector<ValueType, IndexType> &b,
                        const Vector<ValueType, IndexType> &x) {
    if (A.num_cols != b.size)
        throw std::runtime_error{"SingleGPU::ResidualNorm2: A的列数与x维度不一致！"};
    if (b.size != x.size)
        throw std::runtime_error{"SingleGPU::ResidualNorm2: b和x的维度不一致！"};

    Vector<ValueType, IndexType> r = b;
    Residual(&r, A, b, x);
    return Norm2(r);
}

/*
    共轭梯度法求解正定方程组，初始值可由x指向的向量传入。如果details不为nullptr，则会通过
    它返回迭代过程中的残量和时间
*/
template<typename ValueType, typename IndexType>
void CG(Vector<ValueType, IndexType> *x,
        const SparseCSR<ValueType, IndexType> &A,
        const Vector<ValueType, IndexType> &b,
        ValueType error = 1e-4,
        IndexType max_steps = 0,
        std::vector<double> *detail = nullptr,
        DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    IndexType N = b.size;

    // r_0 = b - Ax_0
    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);

    // p_1 = r_0
    Vector<ValueType, IndexType> p = r;

    // r_k^T * r_k and r_{k-1}^T * r_{k-1}
    ValueType rho0 = Dot(r, r);
    ValueType rho1 = 0;
    ValueType norm_r = sqrt(rho0);

    // q_k
    Vector<ValueType, IndexType> q(N);

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    for (IndexType k = 1; k <= max_steps && norm_r > error; ++k) {
        // p_k = r_{k-1} + u_{k-1} * p_{k-1}
        if (k > 1)
            Add(&p, r, p, rho0 / rho1);

        // q_k = A * p_k
        SpMv(&q, A, p);

        // xi_k = (r_{k-1}^T * r_{k-1}) / (p_k^T * q_k)
        ValueType xi = rho0 / Dot(p, q);

        // x_k = x_{k-1} + xi_k * p_k
        Add(x, *x, p, xi);

        // r_k = r{k-1} - xi_k * q_k
        Add(&r, r, q, -xi);

        rho1 = rho0;
        rho0 = Dot(r, r);
        norm_r = sqrt(rho0);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


/*
    BiCG方法，如果求解过程中断，返回true，如果不中断则返回false
    int_bound为中断判断时的边界，必须为一个比较小的正数，当某步计算出的xi的绝对值小于
      此边界时，认为xi等于0，即BiCG方法出现中断，求解失败。
*/
template<typename ValueType, typename IndexType>
bool BiCG(Vector<ValueType, IndexType> *x,
          const SparseCSR<ValueType, IndexType> &A,
          const Vector<ValueType, IndexType> &b,
          ValueType error = 1e-4,
          IndexType max_steps = 0,
          const ValueType int_bound = 1e-10,
          std::vector<double> *detail = nullptr,
          DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    IndexType N = b.size;

    // r_0 = b - Ax_0
    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);

    // 取r*_0 = r_0，以满足r*_0与r_0不正交
    Vector<ValueType, IndexType> r_star = r;

    // p_1 = r_0, p*_1 = r*_0
    Vector<ValueType, IndexType> p = r;
    Vector<ValueType, IndexType> p_star = r_star;

    // r_k^T * r*_k and r_{k-1}^T * r*_{k-1}
    ValueType rho0 = Dot(r, r_star);
    ValueType rho1 = 0;
    ValueType norm_r = Norm2(r);

    // A * p_k and A^T * p_k
    Vector<ValueType, IndexType> q(N);
    Vector<ValueType, IndexType> q_star(N);

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    for (IndexType k = 1; k <= max_steps; ++k) {
        // q_k = A * p_k
        SpMv(&q, A, p);

        // xi_k = (r_{k-1}^T * r*_{k-1}) / (p*_k^T * q_k)
        ValueType xi = rho0 / Dot(p_star, q);

        // xi = 0时算法中断
        if (xi > -int_bound && xi < int_bound)
            return true;

        // x_k = x_{k-1} + xi_k * p_k
        Add(x, *x, p, xi);

        // r_k = r{k-1} - xi_k * q_k
        Add(&r, r, q, -xi);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }

        if (norm_r < error)
            break;

        // q*_k = A^T * p*_k
        SpMTv(&q_star, A, p_star);

        // r*_k = r*_{k-1} - xi_k * A^T * p*_k
        Add(&r_star, r_star, q_star, -xi);

        rho1 = rho0;
        rho0 = Dot(r, r_star);

        // theta_k = (r_k, r*_k) / (r_{k-1}, r*_{k-1})
        ValueType theta = rho0 / rho1;

        // p_{k+1} = r_k + theta_k * p_k
        Add(&p, r, p, theta);

        // p*_{k+1} = r*_k + theta_k * p*_k
        Add(&p_star, r_star, p_star, theta);
    }

    return false;
}


/*
    BiCG方法，系数矩阵A的转置作为一个参数传入
    如果求解过程中断，返回true，如果不中断则返回false
    int_bound为中断判断时的边界，必须为一个比较小的正数，当某步计算出的xi的绝对值小于
      此边界时，认为xi等于0，即BiCG方法出现中断，求解失败。
*/
template<typename ValueType, typename IndexType>
bool BiCG(Vector<ValueType, IndexType> *x,
          const SparseCSR<ValueType, IndexType> &A,
          const SparseCSR<ValueType, IndexType> &At,
          const Vector<ValueType, IndexType> &b,
          ValueType error = 1e-4,
          IndexType max_steps = 0,
          const ValueType int_bound = 1e-10,
          std::vector<double> *detail = nullptr,
          DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};
        
    IndexType N = b.size;

    // r_0 = b - Ax_0
    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);

    // 取r*_0 = r_0，以满足r*_0与r_0不正交
    Vector<ValueType, IndexType> r_star = r;

    // p_1 = r_0, p*_1 = r*_0
    Vector<ValueType, IndexType> p = r;
    Vector<ValueType, IndexType> p_star = r_star;

    // r_k^T * r*_k and r_{k-1}^T * r*_{k-1}
    ValueType rho0 = Dot(r, r_star);
    ValueType rho1 = 0;
    ValueType norm_r = Norm2(r);

    // A * p_k and A^T * p_k
    Vector<ValueType, IndexType> q(N);
    Vector<ValueType, IndexType> q_star(N);

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    for (IndexType k = 1; k <= max_steps; ++k) {
        // q_k = A * p_k
        SpMv(&q, A, p);

        // xi_k = (r_{k-1}^T * r*_{k-1}) / (p*_k^T * q_k)
        ValueType xi = rho0 / Dot(p_star, q);

        // xi = 0时算法中断
        if (xi > -int_bound && xi < int_bound)
            return true;

        // x_k = x_{k-1} + xi_k * p_k
        Add(x, *x, p, xi);

        // r_k = r{k-1} - xi_k * q_k
        Add(&r, r, q, -xi);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }

        if (norm_r < error)
            break;

        // q*_k = A^T * p*_k
        SpMv(&q_star, At, p_star);

        // r*_k = r*_{k-1} - xi_k * A^T * p*_k
        Add(&r_star, r_star, q_star, -xi);

        rho1 = rho0;
        rho0 = Dot(r, r_star);

        // theta_k = (r_k, r*_k) / (r_{k-1}, r*_{k-1})
        ValueType theta = rho0 / rho1;

        // p_{k+1} = r_k + theta_k * p_k
        Add(&p, r, p, theta);

        // p*_{k+1} = r*_k + theta_k * p*_k
        Add(&p_star, r_star, p_star, theta);
    }

    return false;
}


/*
    Richardson迭代最优参数选取，使用幂法估计最大特征值，
    以对角线绝对值最小的元素的绝对值为最小特征值
*/

/*
    计算对角元素绝对值的最小值，不考虑对角元素含有0的情况
*/
template<typename ValueType, typename IndexType>
ValueType FindDiagAbsMin(const SingleGPU::SparseCSR<ValueType, IndexType> &A) {
    const ValueType N = A.num_rows;

    ValueType *diag = nullptr;
    CHECK(cudaMalloc(&diag, sizeof(ValueType) * N));

    LaunchKernel(FindDiag<ValueType, IndexType>, 0, 0,
                 diag, A.values, A.row_offsets, A.column_indices, N);

    int grid_size = 0;
    int block_size = 0;
    LaunchConfig(&grid_size, &block_size);

    ValueType *partial = new ValueType[grid_size];
    ValueType *dev_partial = nullptr;
    CHECK(cudaMalloc(&dev_partial, sizeof(ValueType) * grid_size));

    LaunchKernelWithGrid(MinAbsKernel<ValueType, IndexType>, grid_size, block_size, 0, 0,
                         dev_partial, diag, N);

    CHECK(cudaMemcpy(partial, dev_partial, sizeof(ValueType) * grid_size,
                     cudaMemcpyDeviceToHost));

    ValueType result = std::numeric_limits<ValueType>::max();
    for (int i = 0; i < grid_size; ++i)
        if (result > partial[i])
            result = partial[i];

    delete[] partial;
    CHECK(cudaFree(dev_partial));
    CHECK(cudaFree(diag));

    return result;
}


template<typename ValueType, typename IndexType>
ValueType OptimizePower(const SparseCSR<ValueType, IndexType> &A,
                        IndexType max_steps) {
    IndexType N = A.num_rows;

    Vector<ValueType, IndexType> x1(N);
    Vector<ValueType, IndexType> x2(N);
    Fill(x1, 0.0);
    Fill(x2, 1.0 / N);

    for (IndexType steps = 0; steps < max_steps; ++steps) {
        x1 = x2;

        // x2 = A * x1
        SpMv(&x2, A, x1);

        // 为避免溢出，对x2作归一化处理
        ValueType norm_x2 = Norm2(x2);
        Scale(&x2, x2, 1 / norm_x2);
    }

    Vector<ValueType, IndexType> Av(N);
    SpMv(&Av, A, x2);
    ValueType max_eigenvalue = Dot(x2, Av);

    // 最小特征值以对角线最小元素来估计
    ValueType min_eigenvalue = FindDiagAbsMin(A);

    return 2.0 / (max_eigenvalue + min_eigenvalue);
}


/*
    Richardson迭代，迭代初值通过x传入。omega为最优参数，如果提供一个非零值，则函数
    不再计算，忽略参数power_max_steps，如果提供一个零值，则函数调用OptimizePower计算，
    power_max_steps表示使用幂法求A的最大特征值时迭代多少次，迭代次数越多越精确，
    耗费时间越多，迭代次数过少可能导致求出的参数不能使Richardson迭代法收敛。
    注意：如果需要指定omega为0需要写0.0，否则由0推导出的类型为int而不是ValueType
*/
template<typename ValueType, typename IndexType>
void Richardson(Vector<ValueType, IndexType> *x,
                const SparseCSR<ValueType, IndexType> &A,
                const Vector<ValueType, IndexType> &b,
                ValueType error = 1e-4,
                ValueType omega = 0,
                IndexType power_max_steps = 30,
                IndexType max_steps = 0,
                std::vector<double> *detail = nullptr,
                DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    IndexType N = A.num_rows;

    // 1e-15以下的omega认为是0
    if (-1e-15 < omega && omega < 1e-15)
        omega = OptimizePower(A, power_max_steps);

    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);
    ValueType norm_r = Norm2(r);

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        Add(x, *x, r, omega);
        Residual(&r, A, b, *x);
        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            detail->push_back(norm_r);
        } else {
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


/*
    极小残量Richardson迭代，每步都计算一次最优参数，迭代初值通过x传入
*/
template<typename ValueType, typename IndexType>
void MR(Vector<ValueType, IndexType> *x,
        const SparseCSR<ValueType, IndexType> &A,
        const Vector<ValueType, IndexType> &b,
        ValueType error = 1e-4,
        IndexType max_steps = 0,
        std::vector<double> *detail = nullptr,
        DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    IndexType N = A.num_rows;

    Vector<ValueType, IndexType> r(N);
    Vector<ValueType, IndexType> Ar(N);

    Residual(&r, A, b, *x);
    ValueType norm_r = Norm2(r);

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        // Ar = A * r
        SpMv(&Ar, A, r);

        // omega = (r, Ar) / (Ar, Ar)
        ValueType omega = Dot(r, Ar) / Dot(Ar, Ar);

        // x_{k+1} = x_k + omega * r_k
        Add(x, *x, r, omega);

        // r_{k+1} = r_k - omega * A * r_k
        Add(&r, r, Ar, -omega);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


/*
    Jacobi迭代，迭代初始值通过x传入，这个是标准的Jacobi迭代，通过两次近似值之前的距离来判断
    迭代是否收敛，比下面那个实现要快上少许

template<typename ValueType, typename IndexType>
void Jacobi(Vector<ValueType, IndexType> *x,
            const SparseCSR<ValueType, IndexType> &A,
            const Vector<ValueType, IndexType> &b,
            ValueType error = 1e-4,
            IndexType max_steps = 0) {
    const IndexType N = A.num_rows;

    Vector<ValueType, IndexType> tx(N);
    ValueType delta = std::numeric_limits<ValueType>::infinity();

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    IndexType k = 1;
    while (k++ <= max_steps && delta > error) {
        tx = *x;
        LaunchKernel(JacobiKernel<ValueType, IndexType>, 0, 0, x->data(),
                     tx.data(), A.values, A.row_offsets, A.column_indices, N, b.data());
        delta = Distance(tx, *x);
    }
}
*/

/*
    Jacobi迭代，这个实现做了一些更改，将收敛性的判断条件改为了残量的二范数
*/
template<typename ValueType, typename IndexType>
void Jacobi(Vector<ValueType, IndexType> *x,
            const SparseCSR<ValueType, IndexType> &A,
            const Vector<ValueType, IndexType> &b,
            ValueType error = 1e-4,
            IndexType max_steps = 0,
            std::vector<double> *detail = nullptr,
            DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    const IndexType N = A.num_rows;

    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);
    ValueType norm_r = Norm2(r);

    Vector<ValueType, IndexType> D_1(N);      // A的对角线部分的逆
    LaunchKernel(FindDiagInv<ValueType, IndexType>, 0, 0,
                 D_1.data(), A.values, A.row_offsets, A.column_indices, N);

    Vector<ValueType, IndexType> D_1r(N);
    Vector<ValueType, IndexType> AD_1r(N);

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        // D_1r = D^{-1} * r
        Hadamard(&D_1r, D_1, r);

        // AD_1r = A * D^{-1} * r
        SpMv(&AD_1r, A, D_1r);

        // x_{k+1} = x_k + D^{-1} * r
        Add(x, *x, D_1r, 1.0);

        // r_{k+1} = r_k - A * D^{-1} * r
        Add(&r, r, AD_1r, -1.0);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


/*
    极小残量的Jacobi迭代，迭代初始值通过x传入
*/
template<typename ValueType, typename IndexType>
void MRJacobi(Vector<ValueType, IndexType> *x,
              const SparseCSR<ValueType, IndexType> &A,
              const Vector<ValueType, IndexType> &b,
              ValueType error = 1e-4,
              IndexType max_steps = 0,
              std::vector<double> *detail = nullptr,
              DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    const IndexType N = A.num_rows;

    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);
    ValueType norm_r = Norm2(r);

    Vector<ValueType, IndexType> D_1(N);      // A的对角线部分的逆
    LaunchKernel(FindDiagInv<ValueType, IndexType>, 0, 0,
                 D_1.data(), A.values, A.row_offsets, A.column_indices, N);

    Vector<ValueType, IndexType> D_1r(N);
    Vector<ValueType, IndexType> AD_1r(N);

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        // D_1r = D^{-1} * r
        Hadamard(&D_1r, D_1, r);

        // AD_1r = A * D^{-1} * r
        SpMv(&AD_1r, A, D_1r);

        // alpha = (r^T * A * D^{-1} * r) / ((A * D^{-1} * r)^T (A * D^{-1} * r))
        ValueType alpha = Dot(r, AD_1r) / Dot(AD_1r, AD_1r);

        // x_{k+1} = x_k + alpha * D^{-1} * r
        Add(x, *x, D_1r, alpha);

        // r_{k+1} = r_k - alpha * A * D^{-1} * r
        Add(&r, r, AD_1r, -alpha);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {  // Time
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


/*
    x_{k+1} = x_k + a_k \sqrt{D^{-1}} A \sqrt{D^{-1}} r_k
*/
template<typename ValueType, typename IndexType>
void PreconditionedMRJacobi(Vector<ValueType, IndexType> *x,
                            const SparseCSR<ValueType, IndexType> &A,
                            const Vector<ValueType, IndexType> &b,
                            ValueType error = 1e-4,
                            IndexType max_steps = 0,
                            std::vector<double> *detail = nullptr,
                            DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    const IndexType N = A.num_rows;

    Vector<ValueType, IndexType> sqrt_D_1(N);      // A的对角线部分的逆的平方根
    LaunchKernel(FindDiagInvSqrt<ValueType, IndexType>, 0, 0,
                 sqrt_D_1.data(), A.values, A.row_offsets, A.column_indices, N);

    // 残量
    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);
    Hadamard(&r, sqrt_D_1, r);
    ValueType norm_r = Norm2(r);

    Vector<ValueType, IndexType> sqrt_D_1r(N);             // sqrt{D^{-1}} * r
    Vector<ValueType, IndexType> sqrt_D_1_A_sqrt_D_1r(N);  // sqrt{D^{-1}} * A * sqrt{D^{-1}} * r

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        // sqrt_D_1r = sqrt{D^{-1}} * r
        Hadamard(&sqrt_D_1r, sqrt_D_1, r);

        // sqrt_D_1_A_sqrt_D_1r = sqrt{D^{-1}} * A * sqrt{D^{-1}} * r
        SpMv(&sqrt_D_1_A_sqrt_D_1r, A, sqrt_D_1r);
        Hadamard(&sqrt_D_1_A_sqrt_D_1r, sqrt_D_1, sqrt_D_1_A_sqrt_D_1r);

        ValueType alpha = Dot(r, sqrt_D_1_A_sqrt_D_1r) /
                          Dot(sqrt_D_1_A_sqrt_D_1r, sqrt_D_1_A_sqrt_D_1r);

        // x_{k+1} = x_k + alpha * sqrt{D^{-1}} * r_k
        Add(x, *x, sqrt_D_1r, alpha);

        // r_{k+1} = r_k - alpha * sqrt_{D^{-1}} * A * sqrt{D^{-1}} * r_k
        Add(&r, r, sqrt_D_1_A_sqrt_D_1r, -alpha);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {  // Time
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


/*
    使用原方程残量二范数进行判断而不是预处理后的方程
*/
template<typename ValueType, typename IndexType>
void PreconditionedMRJacobi1(Vector<ValueType, IndexType> *x,
                             const SparseCSR<ValueType, IndexType> &A,
                             const Vector<ValueType, IndexType> &b,
                             ValueType error = 1e-4,
                             IndexType max_steps = 0,
                             std::vector<std::pair<double, double>> *details = nullptr,
                             DetailType details_type = DetailType::Error) {
    const IndexType N = A.num_rows;

    Vector<ValueType, IndexType> sqrt_D_1(N);      // A的对角线部分的逆的平方根
    LaunchKernel(FindDiagInvSqrt<ValueType, IndexType>, 0, 0,
                 sqrt_D_1.data(), A.values, A.row_offsets, A.column_indices, N);

    Vector<ValueType, IndexType> sqrt_D(N);      // A的对角线部分的平方根
    LaunchKernel(FindDiagSqrt<ValueType, IndexType>, 0, 0,
                 sqrt_D.data(), A.values, A.row_offsets, A.column_indices, N);

    // 残量
    Vector<ValueType, IndexType> original_r(N);
    Vector<ValueType, IndexType> r(N);
    Residual(&original_r, A, b, *x);
    Hadamard(&r, sqrt_D_1, original_r);
    ValueType norm_r = Norm2(original_r);

    Vector<ValueType, IndexType> sqrt_D_1r(N);             // sqrt{D^{-1}} * r
    Vector<ValueType, IndexType> sqrt_D_1_A_sqrt_D_1r(N);  // sqrt{D^{-1}} * A * sqrt{D^{-1}} * r

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (details) {
        details->clear();
        details->push_back(std::make_pair(norm_r, 0.0));
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (details)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        // sqrt_D_1r = sqrt{D^{-1}} * r
        Hadamard(&sqrt_D_1r, sqrt_D_1, r);

        // sqrt_D_1_A_sqrt_D_1r = sqrt{D^{-1}} * A * sqrt{D^{-1}} * r
        SpMv(&sqrt_D_1_A_sqrt_D_1r, A, sqrt_D_1r);
        Hadamard(&sqrt_D_1_A_sqrt_D_1r, sqrt_D_1, sqrt_D_1_A_sqrt_D_1r);

        ValueType alpha = Dot(r, sqrt_D_1_A_sqrt_D_1r) /
                          Dot(sqrt_D_1_A_sqrt_D_1r, sqrt_D_1_A_sqrt_D_1r);

        // x_{k+1} = x_k + alpha * sqrt{D^{-1}} * r_k
        Add(x, *x, sqrt_D_1r, alpha);

        // r_{k+1} = r_k - alpha * sqrt_{D^{-1}} * A * sqrt{D^{-1}} * r_k
        Add(&r, r, sqrt_D_1_A_sqrt_D_1r, -alpha);

        Hadamard(&original_r, sqrt_D, r);
        norm_r = Norm2(original_r);

        if (details) {
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            details->push_back(std::make_pair(norm_r, elapsed.count()));
        }
    }
}


/*
    MRS (Minimal Residual Splitting) 方法
*/
template<typename ValueType, typename IndexType>
void MRS(Vector<ValueType, IndexType> *x,
         const SparseCSR<ValueType, IndexType> &A,
         const Vector<ValueType, IndexType> &b,
         const SparseCSR<ValueType, IndexType> &invM,
         ValueType error = 1e-4,
         IndexType max_steps = 0,
         std::vector<double> *detail = nullptr,
         DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    const IndexType N = A.num_rows;

    // 残量
    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);
    ValueType norm_r = Norm2(r);

    Vector<ValueType, IndexType> invM_r(N);    // M^{-1} * r
    Vector<ValueType, IndexType> A_invM_r(N);  // A * M^{-1} * r

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        SpMv(&invM_r, invM, r);
        SpMv(&A_invM_r, A, invM_r);

        ValueType alpha = Dot(r, A_invM_r) / Dot(A_invM_r, A_invM_r);

        // x_{k+1} = x_k + alpha * M^{-1} * r_k
        Add(x, *x, invM_r, alpha);

        // r_{k+1} = r_k - alpha * A * M^{-1} * r_k
        Add(&r, r, A_invM_r, -alpha);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {  // Time
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


/*
    x_{k+1} = x_k + a_k D^{-1} * A * A^T * D^{-1} * r_k
*/
template<typename ValueType, typename IndexType>
void NormalJacobiPreconditionedMR(Vector<ValueType, IndexType> *x,
                                  const SparseCSR<ValueType, IndexType> &A,
                                  const SparseCSR<ValueType, IndexType> &At,
                                  const Vector<ValueType, IndexType> &b,
                                  ValueType error = 1e-4,
                                  IndexType max_steps = 0,
                                  std::vector<double> *detail = nullptr,
                                  DetailType detail_type = DetailType::Error) {
    if (detail_type != DetailType::Error && detail_type != DetailType::Time)
        throw std::runtime_error{"detail type must be one of Time and Error!"};

    const IndexType N = A.num_rows;

    Vector<ValueType, IndexType> D_1(N);      // A的对角线部分的逆
    LaunchKernel(FindDiagInv<ValueType, IndexType>, 0, 0,
                 D_1.data(), A.values, A.row_offsets, A.column_indices, N);

    // 残量
    Vector<ValueType, IndexType> r(N);
    Vector<ValueType, IndexType> temp(N);
    Residual(&temp, A, b, *x);
    SpMv(&r, At, temp);
    Hadamard(&r, D_1, r);
    ValueType norm_r = Norm2(r);

    Vector<ValueType, IndexType> D_1r(N);           // D^{-1} * r
    Vector<ValueType, IndexType> D_1_At_A_D_1r(N);  // D^{-1} * A * A' * D^{-1} * r

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
        }
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    if (detail && detail_type == DetailType::Time)
        start = std::chrono::high_resolution_clock::now();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        // D_1_At_A_D_1r = D^{-1} * A' * A * D^{-1} * r
        Hadamard(&D_1r, D_1, r);
        SpMv(&temp, A, D_1r);
        SpMv(&D_1_At_A_D_1r, At, temp);
        Hadamard(&D_1_At_A_D_1r, D_1, D_1_At_A_D_1r);

        ValueType alpha = Dot(r, D_1_At_A_D_1r) /
                          Dot(D_1_At_A_D_1r, D_1_At_A_D_1r);

        // x_{k+1} = x_k + alpha * D^{-1} * r_k
        Add(x, *x, D_1r, alpha);

        // r_{k+1} = r_k - alpha * D^{-1} * A' * A * D^{-1} * r_k
        Add(&r, r, D_1_At_A_D_1r, -alpha);

        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            // norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            // norm_r = Norm2(r);
            if (detail) {  // Time
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}


} // namespace SingleGPU

#endif // SOLVER_SINGLE_GPU_H
