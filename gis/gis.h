#ifndef GIS_H
#define GIS_H

#include "SparseSystem/SparseSingleGPU.h"
#include "SparseSystem/support.h"

using ValueType = double;
using IndexType = int;

SingleGPU::SparseCSR<ValueType, IndexType> Transpose(
        const SingleGPU::SparseCSR<ValueType, IndexType> &A);

void CG(SingleGPU::Vector<ValueType, IndexType> *x,
        const SingleGPU::SparseCSR<ValueType, IndexType> &A,
        const SingleGPU::Vector<ValueType, IndexType> &b,
        ValueType error = 1e-4,
        int max_steps = 0,
        std::vector<double> *detail = nullptr,
        DetailType detail_type = DetailType::Error);

bool BiCG(SingleGPU::Vector<ValueType, IndexType> *x,
          const SingleGPU::SparseCSR<ValueType, IndexType> &A,
          const SingleGPU::Vector<ValueType, IndexType> &b,
          ValueType error = 1e-4,
          IndexType max_steps = 0,
          const ValueType int_bound = 1e-10,
          std::vector<double> *detail = nullptr,
          DetailType detail_type = DetailType::Error);

bool BiCG(SingleGPU::Vector<ValueType, IndexType> *x,
          const SingleGPU::SparseCSR<ValueType, IndexType> &A,
          const SingleGPU::SparseCSR<ValueType, IndexType> &At,
          const SingleGPU::Vector<ValueType, IndexType> &b,
          ValueType error = 1e-4,
          IndexType max_steps = 0,
          const ValueType int_bound = 1e-10,
          std::vector<double> *detail = nullptr,
          DetailType detail_type = DetailType::Error);

void Richardson(SingleGPU::Vector<ValueType, IndexType> *x,
                const SingleGPU::SparseCSR<ValueType, IndexType> &A,
                const SingleGPU::Vector<ValueType, IndexType> &b,
                ValueType error = 1e-4,
                ValueType omega = 0,
                IndexType power_max_steps = 30,
                IndexType max_steps = 0,
                std::vector<double> *detail = nullptr,
                DetailType detail_type = DetailType::Error);

void MR(SingleGPU::Vector<ValueType, IndexType> *x,
        const SingleGPU::SparseCSR<ValueType, IndexType> &A,
        const SingleGPU::Vector<ValueType, IndexType> &b,
        ValueType error = 1e-4,
        IndexType max_steps = 0,
        std::vector<double> *detail = nullptr,
        DetailType detail_type = DetailType::Error);

void Jacobi(SingleGPU::Vector<ValueType, IndexType> *x,
            const SingleGPU::SparseCSR<ValueType, IndexType> &A,
            const SingleGPU::Vector<ValueType, IndexType> &b,
            ValueType error = 1e-4,
            IndexType max_steps = 0,
            std::vector<double> *detail = nullptr,
            DetailType detail_type = DetailType::Error);

void MRJacobi(SingleGPU::Vector<ValueType, IndexType> *x,
               const SingleGPU::SparseCSR<ValueType, IndexType> &A,
               const SingleGPU::Vector<ValueType, IndexType> &b,
               ValueType error = 1e-4,
               IndexType max_steps = 0,
               std::vector<double> *detail = nullptr,
               DetailType detail_type = DetailType::Error);
                
void JMR(SingleGPU::Vector<ValueType, IndexType> *x,
         const SingleGPU::SparseCSR<ValueType, IndexType> &A,
         const SingleGPU::Vector<ValueType, IndexType> &b,
         ValueType error = 1e-4,
         IndexType max_steps = 0,
         std::vector<double> *detail = nullptr,
         DetailType detail_type = DetailType::Error);

void MRS(SingleGPU::Vector<ValueType, IndexType> *x,
         const SingleGPU::SparseCSR<ValueType, IndexType> &A,
         const SingleGPU::Vector<ValueType, IndexType> &b,
         const SingleGPU::SparseCSR<ValueType, IndexType> &invM,
         ValueType error = 1e-4,
         IndexType max_steps = 0,
         std::vector<double> *detail = nullptr,
         DetailType detail_type = DetailType::Error);

ValueType NormFactor(const SingleGPU::SparseCSR<ValueType, IndexType> &A,
                     const SingleGPU::Vector<ValueType, IndexType> &b,
                     const SingleGPU::Vector<ValueType, IndexType> &x);

#endif // GIS_H
