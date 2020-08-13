#include "cuFoamBiCG.H"

#include "gis/gis.h"
#include "gis/ldu2csr.h"

namespace Foam {

defineTypeNameAndDebug(cuFoamBiCG, 0);

// 对称的(symmetric)
lduMatrix::solver::addsymMatrixConstructorToTable<cuFoamBiCG>
        addBiCGSolverSymMatrixConstructorToTable_;

// 不对称的(asymmetric)
lduMatrix::solver::addasymMatrixConstructorToTable<cuFoamBiCG>
        addBiCGSolverAsymMatrixConstructorToTable_;

} // ~namespace Foam

Foam::cuFoamBiCG::cuFoamBiCG(
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


Foam::solverPerformance Foam::cuFoamBiCG::solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt
) const {
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
    SingleGPU::SparseCSR<ValueType, IndexType> d_At = Transpose(d_A);
    SingleGPU::Vector<ValueType, IndexType> d_b{b};
    SingleGPU::Vector<ValueType, IndexType> d_x{x};

    ValueType norm_factor = NormFactor(d_A, d_b, d_x);

    IndexType max_steps = maxIter_; // 最大迭代步数
    ValueType error = tolerance_;   // 绝对误差，relTol_是相对误差

    std::vector<ValueType> res(max_steps);
    BiCG(&d_x, d_A, d_At, d_b, error * norm_factor, max_steps, 1e-10, &res);

    // 将结果传回OpenFOAM
    d_x.CopyToHost(x);
    std::copy(x.begin(), x.end(), psi.begin());

    solverPerformance solverPerf("cuFoamBiCG", fieldName());
    solverPerf.initialResidual() = res.front();
    solverPerf.finalResidual() = res.back();
    solverPerf.nIterations() = res.size() - 1;

    return solverPerf;
}

