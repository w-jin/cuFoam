#ifndef CUFOAM_MR_JACOBI_H
#define CUFOAM_MR_JACOBI_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamMRJacobi : public lduMatrix::solver {
    cuFoamMRJacobi(const cuFoamMRJacobi &) = delete;
    cuFoamMRJacobi &operator=(const cuFoamMRJacobi &) = delete;

public:
    TypeName("cuFoamMRJacobi");

    cuFoamMRJacobi(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamMRJacobi() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_MR_JACOBI_H
