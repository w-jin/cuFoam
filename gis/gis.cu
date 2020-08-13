#include "gis.h"
#include "SparseSystem/SolverSingleGPU.h"

SingleGPU::SparseCSR<ValueType, IndexType> Transpose(
        const SingleGPU::SparseCSR<ValueType, IndexType> &A) {
    return SingleGPU::Transpose(A);
}

void CG(SingleGPU::Vector<ValueType, IndexType> *x,
        const SingleGPU::SparseCSR<ValueType, IndexType> &A,
        const SingleGPU::Vector<ValueType, IndexType> &b,
        ValueType error,
        int max_steps,
        std::vector<double> *detail,
        DetailType detail_type) {
    SingleGPU::CG(x, A, b, error, max_steps, detail, detail_type);
}

bool BiCG(SingleGPU::Vector<ValueType, IndexType> *x,
          const SingleGPU::SparseCSR<ValueType, IndexType> &A,
          const SingleGPU::Vector<ValueType, IndexType> &b,
          ValueType error,
          IndexType max_steps,
          const ValueType int_bound,
          std::vector<double> *detail,
          DetailType detail_type) {
    return SingleGPU::BiCG(x, A, b, error, max_steps, int_bound, detail, detail_type);
}

bool BiCG(SingleGPU::Vector<ValueType, IndexType> *x,
          const SingleGPU::SparseCSR<ValueType, IndexType> &A,
          const SingleGPU::SparseCSR<ValueType, IndexType> &At,
          const SingleGPU::Vector<ValueType, IndexType> &b,
          ValueType error,
          IndexType max_steps,
          const ValueType int_bound,
          std::vector<double> *detail,
          DetailType detail_type) {
    return SingleGPU::BiCG(x, A, At, b, error, max_steps, int_bound, detail, detail_type);
}

void Richardson(SingleGPU::Vector<ValueType, IndexType> *x,
                const SingleGPU::SparseCSR<ValueType, IndexType> &A,
                const SingleGPU::Vector<ValueType, IndexType> &b,
                ValueType error,
                ValueType omega,
                IndexType power_max_steps,
                IndexType max_steps,
                std::vector<double> *detail,
                DetailType detail_type) {
    SingleGPU::Richardson(x, A, b, error, omega, power_max_steps, max_steps, detail, detail_type);
}

void MR(SingleGPU::Vector<ValueType, IndexType> *x,
        const SingleGPU::SparseCSR<ValueType, IndexType> &A,
        const SingleGPU::Vector<ValueType, IndexType> &b,
        ValueType error,
        IndexType max_steps,
        std::vector<double> *detail,
        DetailType detail_type) {
    SingleGPU::MR(x, A, b, error, max_steps, detail, detail_type);
}

void Jacobi(SingleGPU::Vector<ValueType, IndexType> *x,
            const SingleGPU::SparseCSR<ValueType, IndexType> &A,
            const SingleGPU::Vector<ValueType, IndexType> &b,
            ValueType error,
            IndexType max_steps,
            std::vector<double> *detail,
            DetailType detail_type) {
    SingleGPU::Jacobi(x, A, b, error, max_steps, detail, detail_type);
}

void MRJacobi(SingleGPU::Vector<ValueType, IndexType> *x,
              const SingleGPU::SparseCSR<ValueType, IndexType> &A,
              const SingleGPU::Vector<ValueType, IndexType> &b,
              ValueType error,
              IndexType max_steps,
              std::vector<double> *detail,
              DetailType detail_type) {
    SingleGPU::MRJacobi(x, A, b, error, max_steps, detail, detail_type);
}
                
void JMR(SingleGPU::Vector<ValueType, IndexType> *x,
         const SingleGPU::SparseCSR<ValueType, IndexType> &A,
         const SingleGPU::Vector<ValueType, IndexType> &b,
         ValueType error,
         IndexType max_steps,
         std::vector<double> *detail,
         DetailType detail_type) {
    SingleGPU::PreconditionedMRJacobi(x, A, b, error, max_steps, detail, detail_type);
}

void MRS(SingleGPU::Vector<ValueType, IndexType> *x,
         const SingleGPU::SparseCSR<ValueType, IndexType> &A,
         const SingleGPU::Vector<ValueType, IndexType> &b,
         const SingleGPU::SparseCSR<ValueType, IndexType> &invM,
         ValueType error,
         IndexType max_steps,
         std::vector<double> *detail,
         DetailType detail_type) {
    SingleGPU::MRS(x, A, b, invM, error, max_steps, detail, detail_type);
}


ValueType NormFactor(const SingleGPU::SparseCSR<ValueType, IndexType> &A,
                     const SingleGPU::Vector<ValueType, IndexType> &b,
                     const SingleGPU::Vector<ValueType, IndexType> &x) {
    const IndexType N = A.num_rows;
    ValueType x_bar = SingleGPU::Sum(x) / N;

    SingleGPU::Vector<ValueType, IndexType> X_bar(N);
    SingleGPU::Fill(X_bar, x_bar);

    SingleGPU::Vector<ValueType, IndexType> Y(N);
    SingleGPU::SpMv(&Y, A, x);

    SingleGPU::Vector<ValueType, IndexType> Y_bar(N);
    SingleGPU::SpMv(&Y_bar, A, X_bar);

    // 1e-20为极小的一个数，官方加了这个数
    return SingleGPU::Distance(Y, Y_bar) + SingleGPU::Distance(b, Y_bar) + 1e-20;
}
