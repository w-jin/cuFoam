#ifndef SOLVER_CPU_H
#define SOLVER_CPU_H

#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include <chrono>
#include <limits>
#include <cmath>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "SparseCPU.h"
#include "support.h"

/*
    CPU多线程计算
*/
namespace CPU {

/*
 * return v1^T * v2
 */
template<typename ValueType, typename IndexType>
ValueType Dot(const CPU::Vector<ValueType, IndexType> &v1,
              const CPU::Vector<ValueType, IndexType> &v2) {
    if (v1.size != v2.size)
        throw std::runtime_error{"CPU::Dot: 向量长度不一致！"};

    return tbb::parallel_reduce(
            tbb::blocked_range<IndexType>(0, v1.size), 0.0,
            [&v1, &v2](const tbb::blocked_range<IndexType> &range, ValueType s) -> ValueType {
                for (IndexType i = range.begin(); i != range.end(); ++i)
                    s += v1[i] * v2[i];
                return s;
            },
            [](ValueType s1, ValueType s2) -> ValueType {
                return s1 + s2;
            }
    );
}


/*
 * result = Av
 */
template<typename ValueType, typename IndexType>
void SpMv(Vector <ValueType, IndexType> *result,
          const SparseCSR <ValueType, IndexType> &A,
          const Vector <ValueType, IndexType> &v) {
    if (A.num_cols != v.size)
        throw std::runtime_error{"CPU::SpMv: 矩阵的列数和向量长度不一致！"};

    result->resize(A.num_rows);
    tbb::parallel_for(
            tbb::blocked_range<IndexType>(0, A.num_rows),
            [&A, &v, &result](const tbb::blocked_range<IndexType> &range) -> void {
                for (IndexType i = range.begin(); i != range.end(); ++i) {
                    ValueType t = 0.0;
                    for (IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j)
                        t += A.values[j] * v[A.column_indices[j]];
                    (*result)[i] = t;
                }
            }
    );
}


/*
 * return sqrt(v^T * v)
 */
template<typename ValueType, typename IndexType>
ValueType Norm2(const Vector <ValueType, IndexType> &v) {
    return sqrt(CPU::Dot(v, v));
}


/*
 * return (v1 - v2)^T * (v1 - v2)
 */
template<typename ValueType, typename IndexType>
ValueType Distance(const Vector <ValueType, IndexType> &v1,
                   const Vector <ValueType, IndexType> &v2) {
    if (v1.size != v2.size)
        throw std::runtime_error{"CPU::Distance: 参与运算的两个维量长度不一致！"};

    return sqrt(tbb::parallel_reduce(
            tbb::blocked_range<IndexType>(0, v1.size), 0.0,
            [&v1, &v2](const tbb::blocked_range<IndexType> &range,
                       ValueType s) -> ValueType {
                for (IndexType i = range.begin(); i < range.end(); ++i)
                    s += (v1[i] - v2[i]) * (v1[i] - v2[i]);
                return s;
            },
            [](ValueType s1, ValueType s2) -> ValueType {
                return s1 + s2;
            }
    ));
}


/*
 * result = alpha * v
 */
template<typename ValueType, typename IndexType>
void Scale(CPU::Vector<ValueType, IndexType> *result,
           const CPU::Vector<ValueType, IndexType> &v,
           ValueType alpha) {
    result->resize(v.size);
    tbb::parallel_for(
            tbb::blocked_range<IndexType>(0, v.size),
            [&v, &result, alpha](const tbb::blocked_range<IndexType> &range) -> void {
                for (IndexType i = range.begin(); i != range.end(); ++i)
                    (*result)[i] = alpha * v[i];
            }
    );
}


/*
 * result = v1 + alpha * v2
 */
template<typename ValueType, typename IndexType>
void Add(Vector <ValueType, IndexType> *result,
         const Vector <ValueType, IndexType> &v1,
         const Vector <ValueType, IndexType> &v2,
         ValueType alpha = 1.0) {
    if (v1.size != v2.size)
        throw std::runtime_error{"CPU::Add: 参与运算的两个向量长度不一致！"};

    result->resize(v1.size);
    tbb::parallel_for(
            tbb::blocked_range<IndexType>(0, v1.size),
            [&v1, &v2, &result, alpha](const tbb::blocked_range<IndexType> &range) -> void {
                for (IndexType i = range.begin(); i != range.end(); ++i)
                    (*result)[i] = v1[i] + alpha * v2[i];
            }
    );
}


/*
 * result = alpha * v1 + beta * v2
 */
template<typename ValueType, typename IndexType>
void Add(Vector <ValueType, IndexType> *result,
         const Vector <ValueType, IndexType> &v1,
         const Vector <ValueType, IndexType> &v2,
         ValueType alpha,
         ValueType beta) {
    if (v1.size != v2.size)
        throw std::runtime_error{"CPU::Add: 参与运算的两个向量长度不一致！"};

    result->resize(v1.size);
    tbb::parallel_for(
            tbb::blocked_range<IndexType>(0, v1.size),
            [&v1, &v2, &result, alpha, beta](const tbb::blocked_range<IndexType> &range) -> void {
                for (IndexType i = range.begin(); i != range.end(); ++i)
                    (*result)[i] = alpha * v1[i] + beta * v2[i];
            }
    );
}


/*
 * result[i] = v1[i] * v2[i]
 */
template<typename ValueType, typename IndexType>
void Hadamard(Vector <ValueType, IndexType> *result,
              const Vector <ValueType, IndexType> &v1,
              const Vector <ValueType, IndexType> &v2) {
    if (v1.size != v2.size)
        throw std::runtime_error{"CPU::Add: 参与运算的两个向量长度不一致！"};

    result->resize(v1.size);
    tbb::parallel_for(
            tbb::blocked_range<IndexType>(0, v1.size),
            [&v1, &v2, result](const tbb::blocked_range<IndexType> &range) -> void {
                for (IndexType i = range.begin(); i != range.end(); ++i)
                    (*result)[i] = v1[i] * v2[i];
            }
    );
}


/*
 * 计算残量r = b - Ax，x和r有重叠时行为不定
 */
template<typename ValueType, typename IndexType>
void Residual(Vector <ValueType, IndexType> *r,
              const SparseCSR <ValueType, IndexType> &A,
              const Vector <ValueType, IndexType> &b,
              const Vector <ValueType, IndexType> &x) {
    if (A.num_cols != x.size)
        throw std::runtime_error{"CPU::Residual: A的列数与x维度不一致！"};
    if (A.num_rows != b.size)
        throw std::runtime_error{"CPU::Residual: A的行数和b的维度不一致！"};

    r->resize(b.size);
    tbb::parallel_for(
            tbb::blocked_range<IndexType>(0, A.num_rows),
            [&A, &b, &x, &r](const tbb::blocked_range<IndexType> &range) -> void {
                for (IndexType i = range.begin(); i < range.end(); ++i) {
                    ValueType t = 0.0;
                    for (IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j)
                        t += A.values[j] * x[A.column_indices[j]];
                    (*r)[i] = b[i] - t;
                }
            }
    );
}


/*
 * 计算残量的二范数
 */
template<typename ValueType, typename IndexType>
ValueType ResidualNorm2(const SparseCSR <ValueType, IndexType> &A,
                        const Vector <ValueType, IndexType> &b,
                        const Vector <ValueType, IndexType> &x) {
    if (A.num_cols != b.size)
        throw std::runtime_error{"CPU::Residual: A的列数与x维度不一致！"};
    if (b.size != x.size)
        throw std::runtime_error{"CPU::Residual: b和x的维度不一致！"};

    ValueType rTr = tbb::parallel_reduce(
            tbb::blocked_range<IndexType>(0, A.num_rows), 0.0,
            [&A, &b, &x](const tbb::blocked_range<IndexType> &range,
                         ValueType s) -> ValueType {
                for (IndexType i = range.begin(); i < range.end(); ++i) {
                    ValueType t = 0.0;
                    for (IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j)
                        t += A.values[j] * x[A.column_indices[j]];
                    s += (b[i] - t) * (b[i] - t);
                }
                return s;
            },
            [](ValueType s1, ValueType s2) -> ValueType {
                return s1 + s2;
            }
    );
    return sqrt(rTr);
}


/*
    共轭梯度法求解对称正定方程组，初始值可由x指向的向量传入。如果details不为nullptr，则会通过
    它返回迭代过程中的残量和时间，经过测试，以前的解决方案测得的时间会更短，大约为0.003s，
    但第一个测试的算法将会拥有更大的时间误差，可能是缓存的原因，测试前最好先跑一个算法
*/
template<typename ValueType, typename IndexType>
void CG(Vector <ValueType, IndexType> *x,
        const SparseCSR <ValueType, IndexType> &A,
        const Vector <ValueType, IndexType> &b,
        ValueType error = 1e-4,
        IndexType max_steps = 0,
        std::vector<double> *detail = nullptr,
        DetailType detail_type = DetailType::Error) {
    IndexType N = b.size;

    // r0 = b - Ax0
    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);

    // p1 = r0
    Vector<ValueType, IndexType> p = r;

    // rk^T * rk and rk_1^T * rk_1
    ValueType rho0 = Dot(r, r);
    ValueType rho1 = 0;
    ValueType norm_r = sqrt(rho0);

    // qk
    Vector<ValueType, IndexType> q(N);


    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
            start = std::chrono::high_resolution_clock::now();
        }
    }

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();


    for (IndexType k = 1; k <= max_steps && norm_r > error; ++k) {
        // pk = rk_1 + uk_1 * pk_1
        if (k > 1)
            Add(&p, r, p, rho0 / rho1);

        // qk = A pk
        SpMv(&q, A, p);

        // xik = (rk_1^T * rk_1) / (pk^T * qk)
        ValueType xi = rho0 / Dot(p, q);

        // xk = xk_1 + xik * pk
        Add(x, *x, p, xi);

        // rk = rk_1 - xik * qk
        Add(&r, r, q, -xi);

        rho1 = rho0;
        rho0 = Dot(r, r);
        norm_r = sqrt(rho0);

        if (detail && detail_type == DetailType::Error) {
            norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            norm_r = Norm2(r);
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}

/*
    BiCG方法，系数矩阵A的转置作为一个参数传入
    如果求解过程中断，返回true，如果不中断则返回false
    int_bound为中断判断时的边界，必须为一个比较小的正数，当某步计算出的xi的绝对值小于
    此边界时，认为xi等于0，即BiCG方法出现中断，求解失败。
*/
template<typename ValueType, typename IndexType>
bool BiCG(Vector <ValueType, IndexType> *x,
          const SparseCSR <ValueType, IndexType> &A,
          const SparseCSR <ValueType, IndexType> &At,
          const Vector <ValueType, IndexType> &b,
          ValueType error = 1e-4,
          IndexType max_steps = 0,
          const ValueType int_bound = 1e-10,
          std::vector<double> *detail = nullptr,
          DetailType detail_type = DetailType::Error) {
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


    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
            start = std::chrono::high_resolution_clock::now();
        }
    }

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

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

        if (detail && detail_type == DetailType::Error) {
            norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            norm_r = Norm2(r);
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
    Richardson迭代最优参数选取，使用幂法估计最大特征值，以对角线绝对值
    最小的元素的绝对值来估计最小特征值
*/
template<typename ValueType, typename IndexType>
ValueType OptimizePower(const SparseCSR <ValueType, IndexType> &A,
                        IndexType max_steps) {
    IndexType N = A.num_rows;
    Vector<ValueType, IndexType> x1(N, 0.0);
    Vector<ValueType, IndexType> x2(N, 1.0 / N);

    for (IndexType steps = 0; steps < max_steps; ++steps) {
        x1 = x2;

        // x2 = A * x1
        SpMv(&x2, A, x1);

        // 为避免溢出，对x2作归一化处理
        ValueType norm_x2 = Norm2(x2);
        Scale(&x2, x2, 1 / norm_x2);
    }

    // A * x2 = lambda * x2
    Vector<ValueType, IndexType> Av(N);
    SpMv(&Av, A, x2);
    ValueType max_eigenvalue = Dot(x2, Av);

    // 最小特征值以对角线最小元素来估计
    ValueType min_eigenvalue = tbb::parallel_reduce(
            tbb::blocked_range<IndexType>(0, A.num_rows),
            std::numeric_limits<ValueType>::max(),
            [&A](const tbb::blocked_range<IndexType> &range,
                 ValueType m) -> ValueType {
                for (IndexType i = range.begin(); i < range.end(); ++i) {
                    for (IndexType j = A.row_offsets[i];
                         j < A.row_offsets[i + 1]; ++j) {
                        if (A.column_indices[j] == i) {
                            m = std::min(m, A.values[j]);
                            break;
                        }
                    }
                }
                return m;
            },
            [](ValueType m1, ValueType m2) -> ValueType {
                return std::min(m1, m2);
            }
    );

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
void Richardson(Vector <ValueType, IndexType> *x,
                const SparseCSR <ValueType, IndexType> &A,
                const Vector <ValueType, IndexType> &b,
                ValueType error = 1e-4,
                ValueType omega = 0.0,
                IndexType power_max_steps = 30,
                IndexType max_steps = 0,
                std::vector<double> *detail = nullptr,
                DetailType detail_type = DetailType::Error) {
    IndexType N = A.num_rows;

    // 1e-15以下的omega认为是0
    if (-1e-15 < omega && omega < 1e-15)
        omega = OptimizePower(A, power_max_steps);

    Vector<ValueType, IndexType> r(N);
    Residual(&r, A, b, *x);
    ValueType norm_r = Norm2(r);


    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
            start = std::chrono::high_resolution_clock::now();
        }
    }

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        Add(x, *x, r, omega);
        Residual(&r, A, b, *x);
        norm_r = Norm2(r);

        if (detail && detail_type == DetailType::Error) {
            detail->push_back(norm_r);
        } else if (detail) {
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            detail->push_back(elapsed.count());
        }
    }
}


/*
    极小残量的Richardson迭代，每步都计算一次最优参数，迭代初值通过x传入
*/
template<typename ValueType, typename IndexType>
void MRES_Richardson(Vector <ValueType, IndexType> *x,
                     const SparseCSR <ValueType, IndexType> &A,
                     const Vector <ValueType, IndexType> &b,
                     ValueType error = 1e-4,
                     IndexType max_steps = 0,
                     std::vector<double> *detail = nullptr,
                     DetailType detail_type = DetailType::Error) {
    IndexType N = A.num_rows;

    Vector<ValueType, IndexType> r(N);
    Vector<ValueType, IndexType> Ar(N);

    Residual(&r, A, b, *x);
    ValueType norm_r = Norm2(r);


    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
    if (detail) {
        detail->clear();
        if (detail_type == DetailType::Error)
            detail->push_back(norm_r);
        else { // Time
            detail->push_back(0.0);
            start = std::chrono::high_resolution_clock::now();
        }
    }

    if (max_steps <= 0)
        max_steps = std::numeric_limits<IndexType>::max();

    IndexType k = 1;
    while (k++ <= max_steps && norm_r > error) {
        // Ar = A * r
        SpMv(&Ar, A, r);

        // omega = (r, Ar) / (Ar, Ar)
        ValueType omega = Dot(r, Ar) / Dot(Ar, Ar);

        // x^{k+1} = x^k + omega * r^k
        Add(x, *x, r, omega);

        Residual(&r, A, b, *x);

        if (detail && detail_type == DetailType::Error) {
            norm_r = ResidualNorm2(A, b, *x);
            detail->push_back(norm_r);
        } else {
            norm_r = Norm2(r);
            if (detail) {
                end = std::chrono::high_resolution_clock::now();
                elapsed = end - start;
                detail->push_back(elapsed.count());
            }
        }
    }
}

} // namespace CPU

#endif // SOLVER_CPU_H
