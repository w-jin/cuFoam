#ifndef SPARSE_MULTI_GPU_H
#define SPARSE_MULTI_GPU_H

#include <fstream>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cstdio>

#include <cuda_runtime.h>

#include "SparseSingleGPU.h"
#include "kernels.h"
#include "check.h"

/*
    用于多个GPU计算的类型，假定每个gpu计算能力相同
    todo:全局变量g_num_devices和g_streams不能这样写
    todo:对拥有不同计算能力的gpu系统进行更好的任务划分，可由一个类型为IndexType(int)的函数来进行，
         它可以作为SparseCSR和Vector的一个成员变量取代first_row和first_element
*/
namespace MultiGPU {

// 设备数量
int g_num_devices;

// cuda流，一个设备一个
std::vector<cudaStream_t> g_streams;

/*
    用于创建计算时所需的cuda流，在创建此命名空间内的类型的对象或者调用此命名空间内的函数之前
    必须创建一个Context对象
*/
class Context {
public:
    Context() {
        CHECK(cudaGetDeviceCount(&g_num_devices));
        g_streams.resize(g_num_devices);
        for (int i = 0; i < g_num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            CHECK(cudaStreamCreate(&g_streams[i]));
        }
    }

    ~Context() {
        for (int i = 0; i < g_num_devices; ++i)
            cudaStreamDestroy(g_streams[i]);
    }
};


/*
    分布于各GPU的稀疏矩类型，各GPU上持有连续的多行
    todo:拷贝和移动操作

    ValueType应当是float和double之一，IndexType是整数类型，
    推荐选择int(大多数系统上为32位，最大值为2147483647)或者
    long long(大多数系统上为64位，最大值为9223372036854775807)。
*/
template<typename ValueType = double, typename IndexType = int>
class SparseCSR {
public:
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

        CHECK(cudaGetDeviceCount(&num_devices));

        first_row.resize(num_devices + 1);
        sub_matrices.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_row[i] = (num_rows + num_devices - 1) / num_devices * i;
        first_row[num_devices] = num_rows;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));

            // 将对应的片段截取出来传输到对应的gpu上
            sub_matrices[i] = SingleGPU::SparseCSR<ValueType, IndexType>(
                    h_values + h_row_offsets[first_row[i]],
                    h_row_offsets + first_row[i],
                    h_columns_indices + h_row_offsets[first_row[i]],
                    first_row[i + 1] - first_row[i],
                    num_cols,
                    h_row_offsets[first_row[i + 1]] - h_row_offsets[first_row[i]]
            );

            // 调整row_offsets
            if (i > 0) {
                LaunchKernel(AddKernel3<IndexType, IndexType>, 0, g_streams[i],
                             sub_matrices[i].row_offsets,
                             sub_matrices[i].row_offsets,
                             -h_row_offsets[first_row[i]],
                             sub_matrices[i].num_rows + 1);
            }
        }
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
    SparseCSR(const MultiGPU::SparseCSR<ValueType, IndexType> &A) :
            num_rows{A.num_rows},
            num_cols{A.num_cols},
            num_entries{A.num_entries},
            num_devices{A.num_devices},
            first_row{A.first_row},
            sub_matrices{A.sub_matrices} {}

    // 移动构造
    SparseCSR(MultiGPU::SparseCSR<ValueType, IndexType> &&A) noexcept:
            num_rows{A.num_rows},
            num_cols{A.num_cols},
            num_entries{A.num_entries},
            num_devices{A.num_devices},
            first_row{std::move(A.first_row)},
            sub_matrices{std::move(A.sub_matrices)} {
        A.num_rows = 0;
        A.num_cols = 0;
        A.num_entries = 0;
    }

    // 矩阵行数
    IndexType num_rows;

    // 矩阵列数
    IndexType num_cols;

    // 非零元素个数
    IndexType num_entries;

    // 参与计算的设备数量
    int num_devices;

    // 各设备持有的第一行是原矩阵的第几行，其中
    // first_row[0] = 0,
    // first_row[g_num_devices]=num_rows
    std::vector<IndexType> first_row;

    std::vector<SingleGPU::SparseCSR<ValueType, IndexType>> sub_matrices;
};


/*
    分布于多gpu的向量有两个版本：1、不需要交换数据的，这样的向量只做加减法、内积、求二范数和缩放
    操作，每个gpu上只需要保存一部分数据即可；2、需要交换数据的，这样的向量可能做为矩阵与向量乘法的
    操作数，每个gpu上可能只会持有一部分数据，但必须留出可以容纳其它gpu数据的存储空间，在必要的时候
    调用AllGather来在每个gpu上建立完整的数据。

    这个类规定两种版本的向量共有的操作。
*/
template<typename ValueType = double, typename IndexType = int>
class VectorBase {
public:
    // 改变向量存储空间大小
    virtual void resize(IndexType new_size) = 0;

    // 以某个特定的值填充向量
    virtual void fill(ValueType value) = 0;

    // 将向量复制到内存中
    virtual void CopyToHost(CPU::Vector<ValueType, IndexType> &v) const = 0;

    // 返回第device个设备上的首地址指针，如果是可以同步的向量，它应当返回有效数据段的首地址
    virtual ValueType *data(int device) = 0;

    virtual const ValueType *data(int device) const = 0;

    // 返回第device个设备上的数据长度，如果是可以同步的向量，它应当返回有效数据段的长度
    virtual IndexType length(int device) const = 0;

    // 返回向量的长度
    virtual IndexType length() const = 0;
};


/*
    两种类型的向量的前置声明
*/
template<typename ValueType = double, typename IndexType = int>
class Vector;

template<typename ValueType = double, typename IndexType = int>
class VectorSynchronous;


/*
    分布于多个gpu的向量，每个gpu只持有部分数据，不为其它gpu上的数据保留存储空间
    注意：不能作于矩阵与向量乘法的被乘向量类型
*/
template<typename ValueType, typename IndexType>
class Vector : public VectorBase<ValueType, IndexType> {
public:
    Vector() {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = 0;
        first_element[num_devices] = 0;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>();
        }
    }

    explicit Vector(IndexType N) : size{N} {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = (N + num_devices - 1) / num_devices * i;
        first_element[num_devices] = N;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(length(i));
        }
    }

    explicit Vector(const VectorSynchronous<ValueType, IndexType> &v) {
        this->num_devices = v.num_devices;
        this->first_element = v.first_element;
        this->size = v.size;
        this->values.resize(num_devices);

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(length(i));
            CHECK(cudaMemcpy(data(i), v.data(i),
                             sizeof(ValueType) * length(i), cudaMemcpyDeviceToDevice));
        }
    }

    // 从内存构造，values必须为指向内存中某个地址的指针
    Vector(const ValueType *values, IndexType N) : size{N} {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = (N + num_devices - 1) / num_devices * i;
        first_element[num_devices] = N;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(
                    values + first_element[i], length(i)
            );
        }
    }

    // 从内存中的Vector对象构造
    explicit Vector(const CPU::Vector<ValueType, IndexType> &v) : size{v.size} {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = (size + num_devices - 1) / num_devices * i;
        first_element[num_devices] = size;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(
                    v.data() + first_element[i], length(i)
            );
        }
    }

    // 拷贝构造
    Vector(const Vector<ValueType, IndexType> &v) :
            num_devices{v.num_devices},
            first_element{v.first_element},
            size{v.size} {
        values.resize(num_devices);
        for (int i = 0; i < v.num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            values[i] = v.values[i];
        }
    }

    // 移动构造
    Vector(Vector<ValueType, IndexType> &&v) :
            num_devices{v.num_devices},
            first_element{v.first_element},
            values{std::move(v.values)},
            size{v.size} {
        // 将v设置为空向量，注意保持类型的完整性
        v.size = 0;
        v.values.resize(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            v.first_element[i] = 0;
            v.values[i] = SingleGPU::Vector<ValueType, IndexType>();
        }
        v.first_element[num_devices] = 0;
    }

    // 拷贝赋值操作
    Vector<ValueType, IndexType> &operator=(const Vector<ValueType, IndexType> &v) {
        num_devices = v.num_devices;
        first_element = v.first_element;
        size = v.size;
        values.resize(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            values[i] = v.values[i];
        }

        return *this;
    }

    // 移动赋值操作
    Vector<ValueType, IndexType> &operator=(Vector<ValueType, IndexType> &&v) {
        num_devices = v.num_devices;
        first_element = v.first_element;
        values = std::move(v.values);
        size = v.size;

        // 将v设置为空向量，注意保持类型的完整性
        v.size = 0;
        v.values.resize(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            v.first_element[i] = 0;
            v.values[i] = SingleGPU::Vector<ValueType, IndexType>();
        }
        v.first_element[num_devices] = 0;

        return *this;
    }

    // 改变向量大小
    void resize(IndexType new_size) override {
        if (size != new_size) {
            size = new_size;
            for (int i = 0; i < num_devices; ++i)
                first_element[i] = (size + num_devices - 1) / num_devices * i;
            first_element[num_devices] = size;

            for (int i = 0; i < num_devices; ++i) {
                CHECK(cudaSetDevice(i));
                values[i].resize(length(i));
            }
        }
    }

    // 填充成某个定值
    void fill(ValueType value) override {
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            LaunchKernel(FillKernel<ValueType, IndexType>, 0, g_streams[i],
                         data(i), length(i), value);
        }
    }

    // 传输到内存上
    void CopyToHost(CPU::Vector<ValueType, IndexType> &v) const override {
        v.resize(size);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaStreamSynchronize(g_streams[i]));
            CHECK(cudaMemcpy(v.data() + first_element[i], data(i),
                             sizeof(ValueType) * length(i), cudaMemcpyDeviceToHost));
        }
    }

    // 返回第i个设备上的指针
    ValueType *data(int device) override {
        return values[device].data();
    }

    const ValueType *data(int device) const override {
        return values[device].data();
    }

    // 返回第i个设备上的数据块大小
    IndexType length(int device) const override {
        return first_element[device + 1] - first_element[device];
    }

    // 返回向量长度
    IndexType length() const override {
        return size;
    }

    // 设备数量
    int num_devices;

    // 每个gpu上只保存一部分数据，并且不留出额外的空间
    // 注意：矩阵与向量乘法的结果向量中，first_element必须和稀疏矩阵的first_row完全相等，否则赋值的位置不正确
    std::vector<IndexType> first_element;

    // 数据
    std::vector<SingleGPU::Vector<ValueType, IndexType>> values;

    // 向量长度
    int size;
};


/*
    分布于多个gpu的可同步向量，每个gpu都持有完整的一份数据，在计算过程中可能只有部分有效，
    必要时用AllGather来重新在每个gpu上建立完整的数据

    ValueType应当是float和double之一，IndexType是有符号整数类型，
    推荐选择int(大多数系统上为32位，最大值为2147483647)或者
    long long(大多数系统上为64位，最大值为9223372036854775807)。
    如果不是，可能发生错误。
*/
template<typename ValueType, typename IndexType>
class VectorSynchronous : public VectorBase<ValueType, IndexType> {
public:
    VectorSynchronous() {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);
        this->sync_streams.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = 0;
        first_element[num_devices] = 0;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>();
            CHECK(cudaStreamCreate(&sync_streams[i]));
        }
    }

    explicit VectorSynchronous(IndexType N) : size{N} {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);
        this->sync_streams.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = (N + num_devices - 1) / num_devices * i;
        first_element[num_devices] = N;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(N);
            CHECK(cudaStreamCreate(&sync_streams[i]));
        }
    }

    explicit VectorSynchronous(const Vector<ValueType, IndexType> &v) {
        this->num_devices = v.num_devices;
        this->first_element = v.first_element;
        this->size = v.size;
        this->values.resize(num_devices);
        this->sync_streams.resize(num_devices);

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            CHECK(cudaStreamCreate(&sync_streams[i]));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(size);
            // 补全设备i的数据，放在设备i对应的cuda流上，和后面的计算排在同样的流中
            for (int j = 0; j < num_devices; ++j) {
                CHECK(cudaMemcpyAsync(values[i].data() + first_element[j],
                                      v.data(j), sizeof(ValueType) * length(i),
                                      cudaMemcpyDeviceToDevice,
                                      g_streams[i]));
            }
        }
    }

    // 从内存构造，values必须为指向内存中某个地址的指针
    VectorSynchronous(const ValueType *values, IndexType N) : size{N} {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);
        this->sync_streams.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = (N + num_devices - 1) / num_devices * i;
        first_element[num_devices] = N;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(values, N);
            CHECK(cudaStreamCreate(&sync_streams[i]));
        }
    }

    // 从内存中的Vector对象构造
    explicit VectorSynchronous(const CPU::Vector<ValueType, IndexType> &v) : size{v.size} {
        CHECK(cudaGetDeviceCount(&num_devices));
        this->first_element.resize(num_devices + 1);
        this->values.resize(num_devices);
        this->sync_streams.resize(num_devices);

        for (int i = 0; i < num_devices; ++i)
            first_element[i] = (size + num_devices - 1) / num_devices * i;
        first_element[num_devices] = size;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            values[i] = SingleGPU::Vector<ValueType, IndexType>(v);
            CHECK(cudaStreamCreate(&sync_streams[i]));
        }
    }

    // 拷贝构造
    VectorSynchronous(const VectorSynchronous<ValueType, IndexType> &v) :
            num_devices{v.num_devices},
            first_element{v.first_element},
            size{v.size} {
        values.resize(num_devices);
        sync_streams.resize(num_devices);
        for (int i = 0; i < v.num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            values[i] = v.values[i];
            CHECK(cudaStreamCreate(&sync_streams[i]));
        }
    }

    // 移动构造
    VectorSynchronous(VectorSynchronous<ValueType, IndexType> &&v) noexcept:
            num_devices{v.num_devices},
            first_element{v.first_element},
            values{std::move(v.values)},
            sync_streams{v.sync_streams},
            size{v.size} {
        // 将v设置为空向量，注意保持类型的完整性
        v.size = 0;
        v.values.resize(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            v.first_element[i] = 0;
            v.values[i] = SingleGPU::Vector<ValueType, IndexType>();
            CHECK(cudaStreamCreate(&v.sync_streams[i]));
        }
        v.first_element[num_devices] = 0;
    }

    ~VectorSynchronous() {
        for (int i = 0; i < num_devices; ++i)
            CHECK(cudaStreamDestroy(sync_streams[i]));
    }

    // 拷贝赋值操作
    VectorSynchronous<ValueType, IndexType> &operator=(
            const VectorSynchronous<ValueType, IndexType> &v) {
        num_devices = v.num_devices;
        first_element = v.first_element;
        size = v.size;
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            values[i] = v.values[i];
        }

        return *this;
    }

    // 移动赋值操作
    VectorSynchronous<ValueType, IndexType> &operator=(
            VectorSynchronous<ValueType, IndexType> &&v) noexcept {
        num_devices = v.num_devices;
        first_element = v.first_element;
        values = std::move(v.values);
        size = v.size;

        // 将v设置为空向量，注意保持类型的完整性
        v.size = 0;
        v.values.resize(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            v.first_element[i] = 0;
            v.values[i] = SingleGPU::Vector<ValueType, IndexType>();
        }
        v.first_element[num_devices] = 0;

        return *this;
    }

    VectorSynchronous<ValueType, IndexType> &operator=(
            const Vector<ValueType, IndexType> &v) {
        this->num_devices = v.num_devices;
        this->first_element = v.first_element;
        this->size = v.size;

        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            this->values[i] = SingleGPU::Vector<ValueType, IndexType>(size);
            // 补全设备i的数据，放在设备i对应的cuda流上，和后面的计算排在同样的流中
            for (int j = 0; j < num_devices; ++j) {
                CHECK(cudaMemcpyAsync(values[i].data() + first_element[j],
                                      v.data(j), sizeof(ValueType) * length(i),
                                      cudaMemcpyDeviceToDevice,
                                      g_streams[i]));
            }
        }

        return *this;
    }

    // 改变向量大小, todo:此函数有问题
    void resize(IndexType new_size) override {
        if (size != new_size) {
            size = new_size;
            for (int i = 0; i < num_devices; ++i)
                first_element[i] = (size + num_devices - 1) / num_devices * i;
            first_element[num_devices] = size;

            for (int i = 0; i < num_devices; ++i) {
                CHECK(cudaSetDevice(i));
                values[i].resize(size);
            }
        }
    }

    // 填充成某个定值
    void fill(ValueType value) override {
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            LaunchKernel(FillKernel<ValueType, IndexType>, 0, g_streams[i],
                         values[i].data(), length(), value);
        }
    }

    // 传输到内存上
    void CopyToHost(CPU::Vector<ValueType, IndexType> &v) const override {
        v.resize(size);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaStreamSynchronize(g_streams[i]));
            CHECK(cudaMemcpy(v.data() + first_element[i], data(i),
                             sizeof(ValueType) * length(i), cudaMemcpyDeviceToHost));
        }
    }

    // 返回第i个设备上的指针
    ValueType *data(int device) override {
        return values[device].data() + first_element[device];
    }

    const ValueType *data(int device) const override {
        return values[device].data() + first_element[device];
    }

    // 返回第i个设备上的数据块大小
    IndexType length(int device) const override {
        return first_element[device + 1] - first_element[device];
    }

    // 返回向量长度
    IndexType length() const override {
        return size;
    }

    // 在所有gpu上作一次同步，调用此函数后每个gpu上都有向量的一个完整拷贝，
    // 注意安排同步操作和计算操作的次序，尽量使它们重叠，以达到最优性能
    // todo:测试一下使用更多的流是否可以加速传输过程
    void AllGather() {
        // 设备i将自己的数据发到设备j
        for (int i = 0; i < num_devices; ++i) {
            // 等待第i个设备计算完成，然后把数据发到其它设备
            CHECK(cudaStreamSynchronize(g_streams[i]));
            for (int j = 0; j < num_devices; ++j) {
                // 跳过自己
                if (i == j)
                    continue;
//                if (0 == i)
//                    CHECK(cudaStreamSynchronize(g_streams[j]));
                CHECK(cudaMemcpyAsync(values[j].data() + first_element[i],
                                      values[i].data() + first_element[i],
                                      sizeof(ValueType) * length(i),
                                      cudaMemcpyDeviceToDevice,
//                                      sync_streams[j]));
                                      g_streams[j]));
            }
        }

        /* 方法二
        // 将数据收集到gpu0上来
        for (int i = 1; i < num_devices; ++i) {
            CHECK(cudaMemcpy(values[0].data() + first_element[i],
                             values[i].data() + first_element[i],
                             sizeof(ValueType) * (first_element[i + 1] - first_element[i]),
                             cudaMemcpyDeviceToDevice
            ));
        }

        // 把数据从gpu0分发到其它gpu
        for (int i = 1; i < num_devices; ++i) {
            CHECK(cudaMemcpy(values[i].data(), values[0].data(),
                             sizeof(ValueType) * size, cudaMemcpyDeviceToDevice));
        }
        */

        /* 方法三
        // 设备i去设备j取自己需要的数据
        for (int i = 0; i < num_devices; ++i) {
            // 等待设备i计算完成
            CHECK(cudaStreamSynchronize(g_streams[i]));
            for (int j = 0; j < num_devices; ++j) {
                // 跳过自己
                if (i == j)
                    continue;
                // 等待设备j计算完成
                CHECK(cudaStreamSynchronize(g_streams[j]));
                CHECK(cudaMemcpyAsync(values[i].data() + first_element[j],
                                      values[j].data() + first_element[j],
                                      sizeof(ValueType) * length(j),
                                      cudaMemcpyDeviceToDevice,
                                      sync_streams[j]));
            }
        }
        */
    }

    // 等待同步操作完成
    void WaitAllGather() {
        for (int i = 0; i < num_devices; ++i)
            CHECK(cudaStreamSynchronize(sync_streams[i]));
    }

    // 根据稀疏矩阵的零模型计算需要交换的数据
    void CalculateSwapPattern(const SparseCSR<ValueType, IndexType> &A) {
        swap_pattern.resize(num_devices);

        // 统计哪些列有非零元素
        std::vector<int *> dev_columns_count(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            CHECK(cudaMalloc(&dev_columns_count[i], sizeof(int) * A.sub_matrices[i].num_cols));
            CHECK(cudaMemset(dev_columns_count[i], 0, sizeof(int) * A.sub_matrices[i].num_cols));
            LaunchKernel(ColumnsCountKernel<IndexType>, 0, g_streams[i],
                         dev_columns_count[i],
                         A.sub_matrices[i].column_indices, A.sub_matrices[i].num_entries);
        }

        // 将统计结果传输到内存中
        std::vector<std::vector<int>> column_count(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            CHECK(cudaSetDevice(i));
            column_count[i].resize(A.sub_matrices[i].num_cols);
            CHECK(cudaMemcpy(column_count[i].data(), dev_columns_count[i],
                             sizeof(int) * A.sub_matrices[i].num_cols, cudaMemcpyDeviceToHost));
        }

        // 计算第i个设备需要获取的数据
        for (int i = 0; i < num_devices; ++i) {
            swap_pattern[i].clear();
            // 计算第j段，此段的数据位于第j个设备上
            for (int j = 0; j < num_devices; ++j) {
                // 跳过自己对应的那段
                if (i == j)
                    continue;
                // 寻找所有连续段的起始位置和长度
                IndexType start = -1;
                IndexType end = first_element[j];
                while (end < first_element[j + 1]) {
                    if (column_count[i][end]) {
                        if (start == -1)     // 一个块的起点
                            start = end;
                        ++end;     // 连续非零元素的中间
                    } else {
                        if (start != -1) {   // 一个块的终点
                            IndexType len = end - start;
                            swap_pattern[i].push_back(std::make_tuple(start, len, j));
                            start = -1;      // 此块已结束
                        }
                        ++end;    // 连续零元素的中间
                    }
                }
                if (start != -1) {  // 可能有某块在最后一个位置结束
                    IndexType len = end - start;
                    swap_pattern[i].push_back(std::make_tuple(start, len, j));
                }
            }
        }

        // 释放显存
        for (int i = 0; i < num_devices; ++i)
            CHECK(cudaFree(dev_columns_count[i]));
    }

    // 根据swap_pattern进行同步，仅交换需要的数据，适用于矩阵仅对角线附近有大量零元素的情况
    // 对于一般的稀疏矩阵，需要调用很多次数据量很小的传输操作，性能很差
    void AllGatherWithPattern() {
        if (swap_pattern.empty())
            throw std::runtime_error{
                    "MultiGPU::Vector::SynchronizeWithPattern：未计算swap_pattern！"
            };

        // 传输设备i需要的数据
        for (int i = 0; i < num_devices; ++i) {
            for (int j = 0; j < swap_pattern[i].size(); ++j) {
                IndexType start = std::get<0>(swap_pattern[i][j]);
                IndexType len = std::get<1>(swap_pattern[i][j]);
                int device = std::get<2>(swap_pattern[i][j]);
                CHECK(cudaMemcpyAsync(values[i].data() + start,
                                      values[device].data() + start,
                                      sizeof(ValueType) * len,
                                      cudaMemcpyDeviceToDevice,
                                      g_streams[device]
                ));
            }
        }
        CHECK(cudaDeviceSynchronize());
    }

    // 设备数量
    int num_devices;

    // 若第i个gpu上的数据不是完整的，则[first_element[i], first_element[i+1])
    // 内的数据是有效的。
    // 注意：它必须和SparseCSR的first_row完全相等，否则矩阵与向量乘法
    std::vector<IndexType> first_element;

    // 数据
    std::vector<SingleGPU::Vector<ValueType, IndexType>> values;

    // 向量长度
    int size;

    // synchronize用到的cuda流
    std::vector<cudaStream_t> sync_streams;

    // 做矩阵向量乘法时每个设备上需要哪些来自别的设备的数据
    std::vector<std::vector<std::tuple<IndexType, IndexType, int>>> swap_pattern;
};

} // namespace MultiGPU

#endif // SPARSE_MULTI_GPU_H

