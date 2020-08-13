#ifndef LDU2CSR_H
#define LDU2CSR_H

#include "SparseSystem/SparseCPU.h"
#include "lduMatrix.H"

template<typename ValueType, typename IndexType>
CPU::SparseCSR<ValueType, IndexType> getSparseCSRMat(const Foam::lduMatrix &matrix) {
    IndexType N = matrix.diag().size();   // 单元个数，亦即矩阵阶数
    IndexType M = matrix.lower().size();  // 面数，亦即下三角或上三角非零元素个数

    CPU::SparseCSR<ValueType, IndexType> A(N, N, N + 2 * M);

    const Foam::labelList &losort = matrix.lduAddr().losortAddr();
    const Foam::labelList &losortStart = matrix.lduAddr().losortStartAddr();
    const Foam::labelList &ownerStart = matrix.lduAddr().ownerStartAddr();


    IndexType n = 0;
    for (Foam::label cell = 0; cell < N; ++cell) {
        A.row_offsets[cell] = n;
        for (Foam::label index = losortStart[cell]; index < losortStart[cell + 1]; ++index) {
            Foam::label face = losort[index];
            A.column_indices[n] = matrix.lduAddr().lowerAddr()[face];
            A.values[n] = matrix.lower()[face];
            ++n;
        }

        A.column_indices[n] = cell;
        A.values[n] = matrix.diag()[cell];
        ++n;

        for (Foam::label face = ownerStart[cell]; face < ownerStart[cell + 1]; ++face) {
            A.column_indices[n] = matrix.lduAddr().upperAddr()[face];
            A.values[n] = matrix.upper()[face];
            ++n;
        }
    }
    A.row_offsets[N] = n;

    return A;
    
/*
    IndexType N = matrix.diag().size();   // 单元个数，亦即矩阵阶数
    IndexType M = matrix.lower().size();  // 面数，亦即下三角或上三角非零元素个数
    
    CPU::SparseCSR<ValueType, IndexType> A(N, N);
    
    // 计算各行有多少个非零元素，第0行的保存在row_offsets[1]中.
    // 初始值为1，是因为每行还有一个对角元素
    std::vector<IndexType> row_offsets(N + 1, 1);
    for (IndexType i = 0; i < matrix.lduAddr().lowerAddr().size(); ++i) {
        ++row_offsets[matrix.lduAddr().lowerAddr()[i] + 1];
        ++row_offsets[matrix.lduAddr().upperAddr()[i] + 1];
    }

    row_offsets[0] = 0;
    for (IndexType i = 1; i < row_offsets.size(); ++i)
        row_offsets[i] += row_offsets[i - 1];

    A.row_offsets = row_offsets;

    // 将矩阵元素填入
    A.num_entries = N + 2 * M;
    A.column_indices.resize(N + 2 * M);
    A.values.resize(N + 2 * M);
    
    // 下三角
    for (IndexType i = 0; i < matrix.lower().size(); ++i) {
        IndexType row = matrix.lduAddr().upperAddr()[i];
        IndexType idx = row_offsets[row];
        A.column_indices[idx] = matrix.lduAddr().lowerAddr()[i];
        A.values[idx] = matrix.lower()[i];
        ++row_offsets[row];
    }

    // 对角线
    for (IndexType i = 0; i < N; ++i) {
        IndexType idx = row_offsets[i];
        A.column_indices[idx] = i;
        A.values[idx] = matrix.diag()[i];
        ++row_offsets[i];
    }

    // 上三角
    for (IndexType i = 0; i < matrix.upper().size(); ++i) {
        IndexType row = matrix.lduAddr().lowerAddr()[i];
        IndexType idx = row_offsets[row];
        A.column_indices[idx] = matrix.lduAddr().upperAddr()[i];
        A.values[idx] = matrix.upper()[i];
        ++row_offsets[row];
    }

    return A;
*/
}

#endif // ~LDU2CSR_H
