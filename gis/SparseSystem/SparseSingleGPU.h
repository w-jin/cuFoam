#ifndef SPARSE_SINGLE_GPU_H
#define SPARSE_SINGLE_GPU_H

#include <fstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "SparseCPU.h"
#include "check.h"


/*
    用于单个GPU计算的类型
*/
namespace SingleGPU {

/*
    ValueType应当是float和double之一，IndexType是整数类型，
    推荐选择int(大多数系统上为32位，最大值为2147483647)或者
    long long(大多数系统上为64位，最大值为9223372036854775807)。
*/
template<typename ValueType = double, typename IndexType = int>
class SparseCSR {
public:
    SparseCSR() :
            values{nullptr},
            row_offsets{nullptr},
            column_indices{nullptr},
            num_rows{0},
            num_cols{0},
            num_entries{0} {}

    SparseCSR(IndexType num_rows, IndexType num_cols, IndexType num_entries) {
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->num_entries = num_entries;

        CHECK(cudaMalloc(&values, sizeof(ValueType) * num_entries));
        CHECK(cudaMalloc(&row_offsets, sizeof(IndexType) * (num_rows + 1)));
        CHECK(cudaMalloc(&column_indices, sizeof(IndexType) * num_entries));
    }

    // 以内存中的稀疏矩阵创建显存上的稀疏矩阵，将会复制数据
    SparseCSR(const ValueType *h_values,
              const IndexType *h_row_offsets,
              const IndexType *h_columns_indices,
              IndexType num_rows,
              IndexType num_cols,
              IndexType num_entries) {
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->num_entries = num_entries;

        CHECK(cudaMalloc(&values, sizeof(ValueType) * num_entries));
        CHECK(cudaMalloc(&row_offsets, sizeof(IndexType) * (num_rows + 1)));
        CHECK(cudaMalloc(&column_indices, sizeof(IndexType) * num_entries));

        CHECK(cudaMemcpy(values, h_values,
                         sizeof(ValueType) * num_entries, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(row_offsets, h_row_offsets,
                         sizeof(IndexType) * (num_rows + 1), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(column_indices, h_columns_indices,
                         sizeof(IndexType) * num_entries, cudaMemcpyHostToDevice));
    }

    // 以内存中的稀疏矩阵创建显存上的稀疏矩阵，将会复制数据
    explicit SparseCSR(const CPU::SparseCSR<ValueType, IndexType> &A) :
            SparseCSR(A.values.data(),
                      A.row_offsets.data(),
                      A.column_indices.data(),
                      A.num_rows,
                      A.num_cols,
                      A.num_entries) {}

    // 拷贝构造
    SparseCSR(const SingleGPU::SparseCSR<ValueType, IndexType> &A) :
            num_rows{A.num_rows},
            num_cols{A.num_cols},
            num_entries{A.num_entries} {
        CHECK(cudaMalloc(&values, sizeof(ValueType) * num_entries));
        CHECK(cudaMalloc(&row_offsets, sizeof(IndexType) * (num_rows + 1)));
        CHECK(cudaMalloc(&column_indices, sizeof(IndexType) * num_entries));

        CHECK(cudaMemcpy(values, A.values, sizeof(ValueType) * num_entries,
                         cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(row_offsets, A.row_offsets,
                         sizeof(IndexType) * (num_rows + 1), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(column_indices, A.column_indices,
                         sizeof(IndexType) * num_entries, cudaMemcpyDeviceToDevice));
    }

    // 移动构造
    SparseCSR(SingleGPU::SparseCSR<ValueType, IndexType> &&A)  noexcept :
            num_rows{A.num_rows},
            num_cols{A.num_cols},
            num_entries{A.num_entries} {
        values = A.values;
        row_offsets = A.row_offsets;
        column_indices = A.column_indices;

        A.num_rows = 0;
        A.num_cols = 0;
        A.num_entries = 0;
        A.values = nullptr;
        A.row_offsets = nullptr;
        A.column_indices = nullptr;
    }

    // 拷贝赋值操作
    SparseCSR &operator=(const SingleGPU::SparseCSR<ValueType, IndexType> &A) {
        if (this == &A)
            return *this;

        if (num_rows < A.num_rows) {
            CHECK(cudaFree(row_offsets));
            CHECK(cudaMalloc(&row_offsets, sizeof(IndexType) * (A.num_rows + 1)));
        }
        if (num_entries < A.num_entries) {
            CHECK(cudaFree(values));
            CHECK(cudaFree(column_indices));
            CHECK(cudaMalloc(&values, sizeof(ValueType) * A.num_entries));
            CHECK(cudaMalloc(&column_indices, sizeof(IndexType) * A.num_entries));
        }

        num_rows = A.num_rows;
        num_cols = A.num_cols;
        num_entries = A.num_entries;

        CHECK(cudaMemcpy(values, A.values, sizeof(ValueType) * num_entries,
                         cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(row_offsets, A.row_offsets,
                         sizeof(IndexType) * (num_rows + 1), cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(column_indices, A.column_indices,
                         sizeof(IndexType) * num_entries, cudaMemcpyDeviceToDevice));

        return *this;
    }

    // 移动赋值操作
    SparseCSR &operator=(SingleGPU::SparseCSR<ValueType, IndexType> &&A) noexcept {
        num_rows = A.num_rows;
        num_cols = A.num_cols;
        num_entries = A.num_entries;

        CHECK(cudaFree(values));
        CHECK(cudaFree(row_offsets));
        CHECK(cudaFree(column_indices));

        values = A.values;
        row_offsets = A.row_offsets;
        column_indices = A.column_indices;

        A.num_rows = 0;
        A.num_cols = 0;
        A.num_entries = 0;
        A.values = nullptr;
        A.row_offsets = nullptr;
        A.column_indices = nullptr;

        return *this;
    }

    // 从内存中的稀疏矩阵复制到显存上
    SparseCSR &operator=(const CPU::SparseCSR<ValueType, IndexType> &A) {
        if (num_rows < A.num_rows) {
            CHECK(cudaFree(row_offsets));
            CHECK(cudaMalloc(&row_offsets, sizeof(IndexType) * (A.num_rows + 1)));
        }
        if (num_entries < A.num_entries) {
            CHECK(cudaFree(values));
            CHECK(cudaFree(column_indices));
            CHECK(cudaMalloc(&values, sizeof(ValueType) * A.num_entries));
            CHECK(cudaMalloc(&column_indices, sizeof(IndexType) * A.num_entries));
        }

        num_rows = A.num_rows;
        num_cols = A.num_cols;
        num_entries = A.num_entries;

        CHECK(cudaMemcpy(values, A.values.data(), sizeof(ValueType) * num_entries,
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(row_offsets, A.row_offsets.data(), sizeof(IndexType) * (num_rows + 1),
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(column_indices, A.column_indices.data(), sizeof(IndexType) * num_entries,
                         cudaMemcpyHostToDevice));

        return *this;
    }

    ~SparseCSR() {
        CHECK(cudaFree(values));
        CHECK(cudaFree(row_offsets));
        CHECK(cudaFree(column_indices));
    }

    // 绑定到显存上的数据，不会复制数据，仅将此类用作已有数据的资源句柄，方便调用相关功能接口
    void bind(ValueType *values, IndexType *row_offsets, IndexType *column_indices,
              IndexType num_rows, IndexType num_cols, IndexType num_entries) {
        this->values = values;
        this->row_offsets = row_offsets;
        this->column_indices = column_indices;
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->num_entries = num_entries;
    }

    // 复制到内存中
    void CopyToHost(CPU::SparseCSR<ValueType, IndexType> &A) {
        A.resize(num_rows, num_cols, num_entries);
        CHECK(cudaMemcpy(A.values.data(), values, sizeof(ValueType) * num_entries,
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(A.row_offsets.data(), row_offsets,
                         sizeof(IndexType) * (num_rows + 1), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(A.column_indices.data(), column_indices,
                         sizeof(IndexType) * num_entries, cudaMemcpyDeviceToHost));
    }

    // 非零元素
    ValueType *values;

    // 各行第一个非零元素所在下标
    IndexType *row_offsets;

    // 非零元素所在列
    IndexType *column_indices;

    // 矩阵行数
    IndexType num_rows;

    // 矩阵列数
    IndexType num_cols;

    // 非零元素个数
    IndexType num_entries;

}; // class SparseCSR


/*
    ValueType应当是float和double之一，IndexType是有符号整数类型，
    推荐选择int(大多数系统上为32位，最大值为2147483647)或者
    long long(大多数系统上为64位，最大值为9223372036854775807)。
    如果不是，可能发生错误。
*/
template<typename ValueType = double, typename IndexType = int>
class Vector {
public:
    Vector() : values{nullptr}, size{0}, capacity{0} {}

    explicit Vector(IndexType N) : size{N}, capacity{N} {
        CHECK(cudaMalloc(&values, sizeof(ValueType) * N));
    }

    // 拷贝构造
    Vector(const SingleGPU::Vector<ValueType, IndexType> &v) :
            size{v.size}, capacity{v.size} {
        CHECK(cudaMalloc(&values, sizeof(ValueType) * size));
        CHECK(cudaMemcpy(values, v.values, sizeof(ValueType) * size,
                         cudaMemcpyDeviceToDevice));
    }

    // 移动构造
    Vector(SingleGPU::Vector<ValueType, IndexType> &&v) noexcept:
            size{v.size},
            capacity{v.capacity},
            values{v.values} {
        v.size = 0;
        v.capacity = 0;
        v.values = nullptr;
    }

    // 从内存构造，values必须为指向内存中某个地址的指针
    Vector(const ValueType *values, IndexType N) : size(N), capacity(N) {
        CHECK(cudaMalloc(&this->values, sizeof(ValueType) * N));
        CHECK(cudaMemcpy(this->values, values, sizeof(ValueType) * N,
                         cudaMemcpyHostToDevice));
    }

    // 从内存构造
    explicit Vector(const CPU::Vector<ValueType, IndexType> &v) :
            size{v.size}, capacity{v.size} {
        CHECK(cudaMalloc(&values, sizeof(ValueType) * v.size));
        CHECK(cudaMemcpy(values, v.data(), sizeof(ValueType) * size,
                         cudaMemcpyHostToDevice));
    }

    // 拷贝赋值操作
    Vector<ValueType, IndexType> &operator=(
            const SingleGPU::Vector<ValueType, IndexType> &v) {
        if (this == &v)
            return *this;

        if (capacity < v.size) {
            CHECK(cudaFree(values));
            CHECK(cudaMalloc(&values, sizeof(ValueType) * v.size));
            capacity = v.size;
        }
        size = v.size;
        CHECK(cudaMemcpy(values, v.values, sizeof(ValueType) * size, cudaMemcpyDeviceToDevice));

        return *this;
    }

    // 移动赋值操作
    Vector<ValueType, IndexType> &operator=(SingleGPU::Vector<ValueType, IndexType> &&v)  noexcept {
        CHECK(cudaFree(values));
        values = v.values;
        size = v.size;
        capacity = v.capacity;

        v.values = nullptr;
        v.size = 0;
        v.capacity = 0;

        return *this;
    }

    // 从内存复制
    Vector<ValueType, IndexType> &operator=(const CPU::Vector<ValueType, IndexType> &v) {
        if (capacity < v.size) {
            CHECK(cudaFree(values));
            CHECK(cudaMalloc(&values, sizeof(ValueType) * v.size));
            capacity = v.size;
        }
        size = v.size;
        CHECK(cudaMemcpy(values, v.data(), sizeof(ValueType) * size, cudaMemcpyHostToDevice));

        return *this;
    }

    ~Vector() {
        CHECK(cudaFree(values));
    }

    // 绑定到显存中的数据，values必须为指向显存的某个指针，不会复制数据，仅将此类用作已有
    // 数据的资源句柄，方便调用相关功能接口
    void bind(ValueType *values, int N) {
        capacity = N;
        size = N;
        CHECK(cudaFree(this->values));
        this->values = values;
    }

    // 如果size变小，不释放多余显存，如果size变大，不保留之前的数据，如果需要保留，使用extend
    void resize(IndexType new_size) {
        if (capacity < new_size) {
            CHECK(cudaFree(values));
            CHECK(cudaMalloc(&values, sizeof(ValueType) * new_size));
            capacity = new_size;
        }
        size = new_size;
    }

    // 将显存扩充至new_size个单元，如果new_size<=size效果为resize一致，如果new_size>size，
    // 扩充显存并将先前保存的数据复制过来
    void extend(IndexType new_size) {
        if (capacity < new_size) {
            ValueType *new_values = nullptr;
            CHECK(cudaMalloc(&new_values, sizeof(ValueType) * new_size));
            CHECK(cudaMemcpy(new_values, values, sizeof(ValueType) * size,
                             cudaMemcpyDeviceToDevice));
            CHECK(cudaFree(values));
            values = new_values;
            capacity = new_size;
        }
        size = new_size;
    }

    /*
        释放多余的内存，将会重新分配内存
    */
    void shift_to_fit() {
        if (capacity > size) {
            ValueType *new_values = nullptr;
            CHECK(cudaMalloc(&new_values, sizeof(ValueType) * size));
            CHECK(cudaMemcpy(new_values, values, sizeof(ValueType) * size,
                             cudaMemcpyDeviceToDevice));
            CHECK(cudaFree(values));
            values = new_values;
            capacity = size;
        }
    }

    const ValueType *data() const { return values; }

    ValueType *data() { return values; }

    IndexType length() const { return size; }

    void CopyToHost(CPU::Vector<ValueType, IndexType> &v) const {
        v.resize(size);
        CHECK(cudaMemcpy(v.data(), values, sizeof(ValueType) * size, cudaMemcpyDeviceToHost));
    }

    // 向量元素
    ValueType *values;

    // 向量长度
    IndexType size;

    // 显存容量，当size变小后会有多余空间
    IndexType capacity;

}; // class Vector


/*
    管理计算资源。注意：传递给函数时只能传递非const引用，不能传递值，否则cublas函数调用卡住

struct handle_t {
    handle_t() {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
    }

    ~handle_t() {
        cublasDestroy(cublas_handle);
        cusparseDestroy(cusparse_handle);
    }

    cublasHandle_t cublas_handle;

    cusparseHandle_t cusparse_handle;

    // 阻止用户传值
    handle_t(const handle_t &) = delete;

    handle_t(handle_t &&) = delete;

    handle_t &operator=(const handle_t &) = delete;

    handle_t &operator=(handle_t &&) = delete;
};
*/

} // namespace SingleGPU

#endif // SPARSE_SINGLE_GPU_H

