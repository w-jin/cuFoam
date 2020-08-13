#ifndef CUFOAM_Jacobi_H
#define CUFOAM_Jacobi_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamJacobi : public lduMatrix::solver {
    cuFoamJacobi(const cuFoamJacobi &) = delete;
    cuFoamJacobi &operator=(const cuFoamJacobi &) = delete;

public:
    TypeName("cuFoamJacobi");

    cuFoamJacobi(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamJacobi() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_Jacobi_H
