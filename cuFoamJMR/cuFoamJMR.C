﻿#include "cuFoamJMR.H"

#include "gis/gis.h"
#include "gis/ldu2csr.h"

namespace Foam {

defineTypeNameAndDebug(cuFoamJMR, 0);

// 对称的(symmetric)
lduMatrix::solver::addsymMatrixConstructorToTable<cuFoamJMR>
        addJMRSolverSymMatrixConstructorToTable_;

// 不对称的(asymmetric)
lduMatrix::solver::addasymMatrixConstructorToTable<cuFoamJMR>
        addJMRSolverAsymMatrixConstructorToTable_;

} // ~namespace Foam

Foam::cuFoamJMR::cuFoamJMR(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
) : 
    lduMatrix::solver(
        fieldName,
        matrix,
        coupleIntCoeffs,
        coupleIntCoeffs,
        interfaces,
        solverControls
    )
{}


Foam::solverPerformance Foam::cuFoamJMR::solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt
) const {
    using ValueType = double;
    using IndexType = int;

    IndexType N = matrix().diag().size();

    // 将ldu格式的矩阵转换成csr格式
    CPU::SparseCSR<ValueType, IndexType> A = getSparseCSRMat<ValueType, IndexType>(matrix());

    // 拷贝传入的向量
    CPU::Vector<ValueType, IndexType> b(N);
    std::copy(source.begin(), source.end(), b.begin());

    CPU::Vector<ValueType, IndexType> x(N);
    std::copy(psi.begin(), psi.end(), x.begin());

    // 将数据传输到GPU上
    SingleGPU::SparseCSR<ValueType, IndexType> d_A{A};
    SingleGPU::Vector<ValueType, IndexType> d_b{b};
    SingleGPU::Vector<ValueType, IndexType> d_x{x};

    ValueType norm_factor = NormFactor(d_A, d_b, d_x);

    IndexType max_steps = maxIter_; // 最大迭代步数
    ValueType error = tolerance_;   // 绝对误差，relTol_是相对误差

    std::vector<ValueType> res(max_steps);
    JMR(&d_x, d_A, d_b, error * norm_factor, max_steps, &res);

    // 将结果传回OpenFOAM
    d_x.CopyToHost(x);
    std::copy(x.begin(), x.end(), psi.begin());

    solverPerformance solverPerf("cuFoamJMR", fieldName());
    solverPerf.initialResidual() = res.front();
    solverPerf.finalResidual() = res.back();
    solverPerf.nIterations() = res.size() - 1;
    return solverPerf;
}

