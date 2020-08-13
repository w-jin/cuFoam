#ifndef SPARSE_CPU_H
#define SPARSE_CPU_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>

/*
    用于CPU计算和类型
*/
namespace CPU {

/*
    ValueType应当是float和double之一，IndexType是整数类型，
    推荐选择int(大多数系统上为32位，最大值为2147483647)或者
    long long(大多数系统上为64位，最大值为9223372036854775807)。
*/
template<typename ValueType = double, typename IndexType = int>
class SparseCSR {
public:
    SparseCSR() : num_rows{0}, num_cols{0}, num_entries{0} {}

    SparseCSR(IndexType num_rows, IndexType num_cols, IndexType num_entries = 0) :
            num_rows{num_rows},
            num_cols{num_cols},
            num_entries{num_entries},
            values(num_entries),
            row_offsets(num_rows + 1),
            column_indices(num_entries) {}

    SparseCSR(const SparseCSR<ValueType, IndexType> &A) :
            num_rows{A.num_rows},
            num_cols{A.num_cols},
            num_entries{A.num_entries},
            values{A.values},
            row_offsets{A.row_offsets},
            column_indices{A.column_indices} {}

    SparseCSR(SparseCSR<ValueType, IndexType> &&A) noexcept:
            num_rows{A.num_rows},
            num_cols{A.num_cols},
            num_entries{A.num_entries},
            values{std::move(A.values)},
            row_offsets{std::move(A.row_offsets)},
            column_indices{std::move(A.column_indices)} {
        A.num_rows = 0;
        A.num_entries = 0;
        A.num_cols = 0;
    }

    SparseCSR(IndexType num_rows, IndexType num_cols,
              ValueType *values, IndexType *row_offsets,
              IndexType *column_indices) :
            num_rows{num_rows},
            num_cols{num_cols},
            num_entries{row_offsets[num_rows]},
            values(values, values + row_offsets[num_rows]),
            row_offsets(row_offsets, row_offsets + num_rows + 1),
            column_indices(column_indices, column_indices + row_offsets[num_rows]) {}

    SparseCSR<ValueType, IndexType> &operator=(const SparseCSR<ValueType, IndexType> &A) {
        num_rows = A.num_rows;
        num_cols = A.num_cols;
        num_entries = A.num_entries;
        values = A.values;
        row_offsets = A.row_offsets;
        column_indices = A.column_indices;
        return *this;
    }

    SparseCSR<ValueType, IndexType> &operator=(SparseCSR<ValueType, IndexType> &&A) noexcept {
        num_rows = A.num_rows;
        num_cols = A.num_cols;
        num_entries = A.num_entries;
        values = std::move(A.values);
        row_offsets = std::move(A.row_offsets);
        column_indices = std::move(A.column_indices);
        A.num_rows = 0;
        A.num_cols = 0;
        A.num_entries = 0;
        return *this;
    }

    void resize(IndexType rows, IndexType cols, IndexType entries) {
        num_rows = rows;
        num_cols = cols;
        num_entries = entries;

        values.resize(num_entries);
        row_offsets.resize(num_rows + 1);
        column_indices.resize(num_entries);
    }

    /*
        从数据文件中加载矩阵。文件前3*sizeof(IndexType)个字节每4个字节分别为矩阵行数、列数、非零元素个数。
        注意：加载一个文件前必须先知道它保存的数据类型，亦是此类的对象声明时需要提供的类型
        注意：在不同字节序的机器上数据文件可能不能通用。
    */
    void load(const std::string &filename) {
        std::ifstream infile(filename, std::ios::binary);

        if (!infile)
            throw std::runtime_error{std::string("SparseCSR::load can't read file ") + filename};

        // 矩阵规模，行数、列数、非零元素个数
        infile.read(reinterpret_cast<char *>(&num_rows), sizeof(IndexType));
        infile.read(reinterpret_cast<char *>(&num_cols), sizeof(IndexType));
        infile.read(reinterpret_cast<char *>(&num_entries), sizeof(IndexType));

        // 分配空间
        values.resize(num_entries);
        row_offsets.resize(num_rows + 1);
        column_indices.resize(num_entries);

        // 读入矩阵
        infile.read(reinterpret_cast<char *>(values.data()), num_entries * sizeof(ValueType));
        infile.read(reinterpret_cast<char *>(row_offsets.data()), num_rows * sizeof(IndexType));
        infile.read(reinterpret_cast<char *>(column_indices.data()), num_entries * sizeof(IndexType));

        // 哨兵
        row_offsets[num_rows] = num_entries;
    }

    /*
        将矩阵写入数据文件。写入的文件前3*sizeof(IndexType)个字节为矩阵行数、列数、非零元素个数，
        由save函数创建的数据文件可以被load函数读取。
        注意：在不同字节序的机器上数据文件可能不能通用。
    */
    void save(const std::string &filename) const {
        std::ofstream outfile(filename, std::ios::binary);

        // 矩阵规模，行数、列数、非零元素个数
        outfile.write(reinterpret_cast<const char *>(&num_rows), sizeof(IndexType));
        outfile.write(reinterpret_cast<const char *>(&num_cols), sizeof(IndexType));
        outfile.write(reinterpret_cast<const char *>(&num_entries), sizeof(IndexType));

        // 写入矩阵
        outfile.write(reinterpret_cast<const char *>(values.data()), num_entries * sizeof(ValueType));
        outfile.write(reinterpret_cast<const char *>(row_offsets.data()), num_rows * sizeof(IndexType));
        outfile.write(reinterpret_cast<const char *>(column_indices.data()), num_entries * sizeof(IndexType));
    }

    /*
        从RB(Rutherford Boeing)格式的文件中读取稀疏矩阵。
        RB格式参见：
            [1] http://people.math.sc.edu/Burkardt/data/rb/rb.html
            [2] Iain Duff, Roger Grimes, John Lewis,
                User's Guide for the Harwell-Boeing Sparse Matrix Collection,
                Technical Report TR/PA/92/86, CERFACS, October 1992.
            [3] Iain Duff, Roger Grimes, John Lewis,
                The Rutherford-Boeing Sparse Matrix Collection,
                Technical Report RAL-TR-97-031, Rutherford Appleton Laboratory, 1997.

        注意：RB格式将数据保存于文本文件之中，因此不存在模板值的问题。
        注意：此程序暂时只处理实矩阵，若给定矩阵为复矩阵，将抛出异常
        注意：此程序暂时不考虑单元格式
    */
    void readRB(const std::string &filename) {
        std::ifstream infile(filename);

        if (!infile)
            throw std::runtime_error{std::string("SparseCSR::readRB can't read file ") + filename};

        // 前两行暂时不用
        std::string line1, line2;
        std::getline(infile, line1);
        std::getline(infile, line2);

        // 第三行第一段为矩阵类型，含有三个字符，简要信息如下
        // 第一个：r - 实矩阵，c - 复矩阵，i - 整数矩阵，p - 零模型
        // 第二个：s - 对称，u - 不对称，h - 埃尔米特，z - 反对称，r - 方阵
        // 第三个：a - 列压缩，e - 单元格式
        std::string type;
        infile >> type;
        if (type[0] != 'r' && type[0] != 'i')
            throw std::runtime_error{"SparseCSR::readRB不支持实数矩阵以外的矩阵！"};
        if (type[2] != 'a')
            throw std::runtime_error{"SparseCSR::readRB不支持elemental format！"};

        // 第三行第二、三、四段分别为矩阵行数、列数和非零元个数
        IndexType rows = 0;
        IndexType cols = 0;
        IndexType entries = 0;
        infile >> rows >> cols >> entries;

        // 第三行第五段暂时未使用，应当保持为0
        int unused = 0;
        infile >> unused;

        // 第四行为每个数据的宽度以及每行有多少个数据，c++格式化io中不需要这些信息
        std::string line4;
        std::getline(infile, line4);   // 指针在第三行，先消耗掉
        std::getline(infile, line4);

        // 第五行开始为矩阵数据，RB采用csc格式保存稀疏矩阵，当矩阵为对称矩阵或者反对称矩阵时，
        // 文件中只保存了一半的数据，需要补齐另一半元素，若矩阵不是对称矩阵，则需要做转置
        // 注意：RB格式采用1-based存储数据，应该将其转换成0-based

        // 先当成转置读进来
        SparseCSR<ValueType, IndexType> A(cols, rows, entries);

        IndexType temp = 0;
        for (IndexType i = 0; i <= A.num_rows; ++i) {
            infile >> temp;
            A.row_offsets[i] = temp - 1;
        }

        for (IndexType i = 0; i < A.num_entries; ++i) {
            infile >> temp;
            A.column_indices[i] = temp - 1;
        }

        for (IndexType i = 0; i < A.num_entries; ++i) {
            infile >> A.values[i];
        }

        // 如果不对称，将其转置
        if (type[1] == 'u' || type[1] == 'r') {
            *this = A.transpose();
        }
        // 如果对称，需要补齐另一半元素
        if (type[1] == 's') {
            // 先计算row_offsets
            num_rows = num_cols = rows;
            row_offsets = A.row_offsets;
            for (IndexType i = 0; i < A.num_rows; ++i) {
                for (IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
                    if (i == A.column_indices[j])
                        continue;
                    for (IndexType k = A.column_indices[j] + 1; k <= A.num_rows; ++k)
                        ++row_offsets[k];
                }
            }

            // 计算values和column_indices
            num_entries = row_offsets[num_rows];
            values.resize(num_entries);
            column_indices.resize(num_entries);

            std::vector<IndexType> curr = row_offsets;  // 记录填充到哪个位置
            for (IndexType i = 0; i < A.num_rows; ++i) {
                for (IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
                    values[curr[i]] = A.values[j];
                    column_indices[curr[i]] = A.column_indices[j];
                    ++curr[i];
                    if (i != A.column_indices[j]) {
                        values[curr[A.column_indices[j]]] = A.values[j];
                        column_indices[curr[A.column_indices[j]]] = i;
                        ++curr[A.column_indices[j]];
                    }
                }
            }
        }
        // 反对称，需要补齐另一半元素的相反数
        if (type[1] == 'z') {
            // 先计算row_offsets
            num_rows = num_cols = rows;
            row_offsets = A.row_offsets;
            for (IndexType i = 0; i < A.num_rows; ++i) {
                for (IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
                    if (i == A.column_indices[j])
                        continue;
                    for (IndexType k = A.column_indices[j] + 1; k <= A.num_rows; ++k)
                        ++row_offsets[k];
                }
            }

            // 计算values和column_indices
            num_entries = row_offsets[num_rows];
            values.resize(num_entries);
            column_indices.resize(num_entries);

            std::vector<IndexType> curr = row_offsets;  // 记录填充到哪个位置
            for (IndexType i = 0; i < A.num_rows; ++i) {
                for (IndexType j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
                    values[curr[i]] = A.values[j];
                    column_indices[curr[i]] = A.column_indices[j];
                    ++curr[i];
                    if (i != A.column_indices[j]) {
                        values[curr[A.column_indices[j]]] = -A.values[j];  // 对称位置上填充相反数
                        column_indices[curr[A.column_indices[j]]] = i;
                        ++curr[A.column_indices[j]];
                    }
                }
            }

            // 读入的是原矩阵的转置，需要再对矩阵作一次转置，A^T=-A
            for (IndexType i = 0; i < A.num_entries; ++i)
                A.values[i] = -A.values[i];
        }
    }

    /*
        根据后缀名来决定以什么格式读取数据
        .rb - Rutherford Boeing格式
        .mtx - Matrix Market，暂时不支持，未来可能支持
        other - 程序独有格式，不能用于不同字节序的机器
    */
    void read(const std::string &filename) {
        std::string::size_type n = filename.size();
        if (n >= 3 && filename.substr(n - 3, 3) == ".rb")
            readRB(filename);
        else
            load(filename);
    }

    /*
        将稠密矩阵转换为稀疏矩阵
        data - 数据首指针，从本身的类型转换成ValueType类型
        M - 矩阵行数
        N - 矩阵列数
        lda - 一行需要跳过多少个元素，0则为紧密存储，一行跳过N个元素
    */
    template<typename InValueType>
    void fromDense(const InValueType *data, int M, int N, int lda = 0) {
        if (0 == lda)
            lda = N;

        // 清理已有数据
        num_rows = M;
        num_cols = N;
        num_entries = 0;
        row_offsets.resize(M + 1);
        column_indices.clear();
        values.clear();

        for (int i = 0; i < M; ++i) {
            row_offsets[i] = num_entries;
            for (int j = 0; j < N; ++j) {
                if (0 == data[i * lda + j])
                    continue;
                values.push_back(data[i * lda + j]);
                column_indices.push_back(j);
                ++num_entries;
            }
        }
        row_offsets[M] = num_entries;
    }

    /*
        将稀疏矩阵转换为稠密矩阵。
        注意：data的空间必须提前分配好，典型的空间大小为sizeof(OutValueType) *
             num_rows * num_cols个字节
        data - 矩阵元素存储的首地址，各个元素从ValueType转换成它本身的类型
        lda - 一行跳过多少个元素，0则为紧密储存，一行跳过num_cols个元素，它的值不要少于num_cols
    */
    template<typename OutValueType>
    void toDense(OutValueType *data, int lda = 0) const {
        if (0 == lda)
            lda = num_cols;
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j)
                data[i * lda + j] = 0;
            for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j)
                data[i * lda + column_indices[j]] = values[j];
        }
    }

    /*
        转置，也可以看作csr和csc格式之间的转换。时间复杂度O(nnz + max{rows, cols})
        另一个方案是转成coo格式，交换行和列，按列号排序，同列按行号排序，再转回csr。时间复杂度O(nnz * log(nnz))。
    */
    SparseCSR<ValueType, IndexType> transpose() const {
        SparseCSR<ValueType, IndexType> At(num_cols, num_rows, num_entries);
        std::fill(At.row_offsets.begin(), At.row_offsets.end(), 0);

        // 计算A^T每行有多少个非零元素
        for (IndexType i = 0; i < num_entries; ++i)
            ++(At.row_offsets[column_indices[i]]);

        // 计算A^T每行首个非零元素偏移量
        for (IndexType i = 0, sum = 0; i < At.num_rows; ++i) {
            IndexType temp = At.row_offsets[i];
            At.row_offsets[i] = sum;
            sum += temp;
        }
        At.row_offsets[At.num_rows] = At.num_entries;

        // 使用curr记录各行已填充到哪个位置
        std::vector<IndexType> curr = At.row_offsets;

        // 将非零元素复制到A^T中
        for (IndexType i = 0; i < num_rows; ++i) {
            for (IndexType j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
                IndexType col = column_indices[j];
                At.values[curr[col]] = values[j];
                At.column_indices[curr[col]] = i;
                ++(curr[col]);
            }
        }

        return At;
    }

    /*
        计算稀疏矩阵的对角线部分
    */
    SparseCSR<ValueType, IndexType> diagonal() const {
        if (num_rows != num_cols)
            throw std::runtime_error{"SparseCSR::diagonal不能处理非方阵"};

        SparseCSR<> D(num_rows, num_cols, num_rows);
        D.row_offsets[0] = 0;
        for (int i = 0; i < num_rows; ++i) {
            int j = row_offsets[i];
            while (j < row_offsets[i + 1] && column_indices[j] < i)
                ++j;
            // 对角元素为0
            if (j == row_offsets[i + 1] || column_indices[j] != i)
                continue;
            D.values[i] = values[j];
            D.column_indices[i] = i;
            D.row_offsets[i + 1] = i + 1;
        }

        return D;
    }

    /*
        计算稀疏矩阵的Jacobi预处理子，即inv(D)。
    */
    SparseCSR<ValueType, IndexType> JacobiPrecondition() const {
        if (num_rows != num_cols)
            throw std::runtime_error{"SparseCSR::JacobiPrecondition不能处理非方阵"};

        SparseCSR<> invD(num_rows, num_cols, num_rows);
        invD.row_offsets[0] = 0;
        for (int i = 0; i < num_rows; ++i) {
            int j = row_offsets[i];
            while (j < row_offsets[i + 1] && column_indices[j] < i)
                ++j;
            if (j == row_offsets[i + 1] || column_indices[j] != i)
                throw std::runtime_error{"SparseCSR::JacobiPrecondition: D is not invertible"};
            invD.values[i] = 1 / values[j];
            invD.column_indices[i] = i;
            invD.row_offsets[i + 1] = i + 1;
        }

        return invD;
    }

    /*
        计算稀疏矩阵的Gauss-Seidel预处理子，即inv(D-L)。首先D-L是一个下三角矩阵，对下三角求
        逆可以这样考虑：
        记
    */
    SparseCSR<ValueType, IndexType> GaussPrecondition() const {
        if (num_rows != num_cols)
            throw std::runtime_error{"SparseCSR::GaussPrecondition不能处理非方阵"};

        // 先按转置存储inv(D-L)，即按列计算inv(D-L)，而将列存储为行
        SparseCSR<ValueType, IndexType> inv_D_L(num_cols, num_rows);
        inv_D_L.row_offsets[0] = 0;
        for (int j = 0; j < num_cols; ++j) {  // 计算第j列，存储为第j行
            // 严格上三角部分全部为0，从对角线元素开始计算
            int k = row_offsets[j];
            while (k < row_offsets[j + 1] && column_indices[k] < j)
                ++k;
            if (k == row_offsets[j + 1] || column_indices[k] != j)
                throw std::runtime_error{"SparseCSR::GaussPrecondition: D-L is not invertible"};
            inv_D_L.values.push_back(1 / values[k]);
            inv_D_L.column_indices.push_back(j);
            ++inv_D_L.num_entries;

            // 计算严格下三角部分
            for (int i = j + 1; i < num_rows; ++i) {
                double t = 0;
                int k1 = row_offsets[i];
                int k2 = inv_D_L.row_offsets[j];
                while (k1 < row_offsets[i + 1] && k2 < inv_D_L.num_entries) {
                    if (column_indices[k1] == inv_D_L.column_indices[k2]) {
                        t += values[k1] * inv_D_L.values[k2];
                        ++k1;
                        ++k2;
                    } else if (column_indices[k1] < inv_D_L.column_indices[k2]) {
                        ++k1;
                    } else {
                        ++k2;
                    }
                }
                if (0 == t)   // 0元素
                    continue;
                // 找对角线元素
                while (k1 < row_offsets[i + 1] && column_indices[k1] < i)
                    ++k1;
                if (k1 == row_offsets[i + 1] || column_indices[k1] != i)
                    throw std::runtime_error{"SparseCSR::GaussPrecondition: D-L is not invertible"};
                inv_D_L.values.push_back(t / values[k1]);
                inv_D_L.column_indices.push_back(i);
                ++inv_D_L.num_entries;
            }
            // 一列计算完毕，即此列已经存储为一行
            inv_D_L.row_offsets[j + 1] = inv_D_L.num_entries;
        }

        return inv_D_L.transpose();
    }

    // 以coo格式将矩阵输出到标准输出
    void display() const {
        // 行号宽度
        int i_width = 5;
        if (num_rows > 1e5)
            i_width = 10;
        if (num_rows > 1e10)
            i_width = 15;
        // 列号宽度
        int j_width = 5;
        if (num_cols > 1e5)
            j_width = 10;
        if (num_cols > 1e10)
            j_width = 15;
        // 设置左对齐及浮点精度为10位有效数字
        std::cout << std::left << std::setprecision(10);

        std::cout << "sparse matrix <" << num_rows << ", " << num_cols
                  << "> with " << num_entries << " entries:\n";
        std::cout << std::setw(i_width) << "row"
                  << std::setw(j_width) << "col"
                  << "values" << "\n";
        std::cout << "--------------------------------\n";
        for (IndexType i = 0; i < num_rows; ++i) {
            for (IndexType j = row_offsets[i]; j < row_offsets[i + 1]; ++j)
                std::cout << std::setw(i_width) << i
                          << std::setw(j_width) << column_indices[j]
                          << values[j] << "\n";
        }

        // 将std::cout的设置还原
        std::cout << std::right << std::setprecision(6);
    }

    // 以稠密矩阵格式将矩阵输出到标准输出
    void PrintDense(IndexType max_rows = std::numeric_limits<IndexType>::max(),
                    IndexType max_cols = std::numeric_limits<IndexType>::max()) const {
        std::cout << "sparse matrix <" << num_rows << ", " << num_cols
                  << "> with " << num_entries << " entries:\n";

        auto width = std::cout.precision() + 7;
        for (IndexType i = 0; i < num_rows && i < max_rows; ++i) {
            int j = 0;
            for (IndexType curr_nz = row_offsets[i]; curr_nz < row_offsets[i + 1]; ++curr_nz) {
                while (j < column_indices[curr_nz] && j < max_cols) {
                    std::cout << std::setw(width) << 0.0;
                    ++j;
                }
                if (j == max_cols)
                    break;
                std::cout << std::setw(width) << values[curr_nz];
                ++j;
            }
            while (j < num_cols && j < max_cols) {
                std::cout << std::setw(width) << 0.0;
                ++j;
            }
            std::cout << "\n";
        }
    }


    // 返回某些行组成的子稀疏矩阵，行的范围为[first_row, last_row)
    SparseCSR<ValueType, IndexType> subMatrix(IndexType first_row,
                                              IndexType last_row) const {
        if (first_row < 0 || last_row > num_rows || last_row < first_row)
            throw std::runtime_error{"SparseCSR::subMatrix：子矩阵下标范围不正确！"};

        SparseCSR<ValueType, IndexType> sub_mat(last_row - first_row, num_cols,
                                                row_offsets[last_row] - row_offsets[first_row]);
        std::copy(values.begin() + row_offsets[first_row], values.begin() + row_offsets[last_row],
                  sub_mat.values.begin());
        std::copy(column_indices.begin() + row_offsets[first_row],
                  column_indices.begin() + row_offsets[last_row], sub_mat.column_indices.begin());
        for (IndexType i = 0; i <= last_row - first_row; ++i)
            sub_mat.row_offsets[i] = row_offsets[i + first_row] - row_offsets[first_row];

        return sub_mat;
    }


    // 矩阵行数
    IndexType num_rows;

    // 矩阵列数
    IndexType num_cols;

    // 非零元素个数
    IndexType num_entries;

    // 非零元素
    std::vector<ValueType> values;

    // 各行第一个非零元素所在下标
    std::vector<IndexType> row_offsets;

    // 非零元素所在列
    std::vector<IndexType> column_indices;

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
    Vector() : size{0} {}

    explicit Vector(IndexType n) :
            size{n},
            values(n) {}

    Vector(IndexType n, ValueType x) :
            size{n},
            values(n, x) {}

    Vector(const Vector<ValueType, IndexType> &v) :
            size{v.size},
            values{v.values} {}

    Vector(Vector<ValueType, IndexType> &&v) noexcept:
            size{v.size},
            values{std::move(v.values)} {
        v.size = 0;
    }

    Vector<ValueType, IndexType> &operator=(const Vector<ValueType, IndexType> &v) {
        if (this != &v)
            return *this;
        size = v.size;
        values = v.values;
        return *this;
    }

    Vector<ValueType, IndexType> &operator=(Vector<ValueType, IndexType> &&v) noexcept {
        if (this == &v)
            return *this;
        size = v.size;
        values = std::move(v.values);
        v.size = 0;
        return *this;
    }

    void load(const std::string &filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile)
            throw std::runtime_error{std::string("Vector::load can't read file ") + filename};

        infile.read(reinterpret_cast<char *>(&size), sizeof(IndexType));
        values.resize(size);
        infile.read(reinterpret_cast<char *>(values.data()), size * sizeof(ValueType));
    }

    // 从matrix market格式的文件中读取向量数据
    void readMTX(const std::string &filename) {
        std::ifstream infile(filename);
        if (!infile)
            throw std::runtime_error{std::string("Vector::readMTX can't read file ") + filename};

        // 消耗第一行
        std::string line;
        std::getline(infile, line);

        // 向量大小
        infile >> this->size;

        // 消耗第二行
        std::getline(infile, line);

        // 读入数据
        this->resize(this->size);
        for (int i = 0; i < this->size; ++i)
            infile >> this->values[i];
    }

    void save(const char *filename) const {
        std::ofstream outfile(filename, std::ios::binary);
        outfile.write(reinterpret_cast<const char *>(&size), sizeof(IndexType));
        outfile.write(reinterpret_cast<const char *>(values.data()), size * sizeof(ValueType));
    }

    /*
    * 将向量设置为[v, v, v, ...]的形式。
    */
    void fill(ValueType v) {
        for (IndexType i = 0; i < size; ++i)
            values[i] = v;
    }

    void display() const {
        for (IndexType i = 0; i < size; ++i)
            printf("%lf\n", values[i]);
    }

    void resize(IndexType new_size) {
        values.resize(new_size);
        size = new_size;
    }

    void push_back(ValueType &v) {
        values.push_back(v);
        ++size;
    }

    // 释放多余的内存
    void shift_to_fit() {
        values.resize(size);
        values.shift_to_fit();
    }

    ValueType &operator[](IndexType i) { return values[i]; }

    const ValueType &operator[](IndexType i) const { return values[i]; }

    typename std::vector<ValueType>::iterator begin() { return values.begin(); }

    typename std::vector<ValueType>::const_iterator begin() const { return values.begin(); }

    typename std::vector<ValueType>::iterator end() { return values.end(); }

    typename std::vector<ValueType>::const_iterator end() const { return values.end(); }

    const ValueType *data() const { return values.data(); }

    ValueType *data() { return values.data(); }


    // 向量长度
    IndexType size;

    // 向量元素
    std::vector<ValueType> values;

}; // class Vector


/*
    返回一个单位矩阵
*/
template<typename ValueType, typename IndexType>
SparseCSR<ValueType, IndexType> eye(int N) {
    SparseCSR<ValueType, IndexType> I(N, N, N);
    for (int i = 0; i <= N; ++i)
        I.row_offsets[i] = i;
    for (int i = 0; i < N; ++i) {
        I.values[i] = 1;
        I.column_indices[i] = i;
    }
    return I;
}

} // namespace CPU

#endif // SPARSE_CPU_H

