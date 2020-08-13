#ifndef CUFOAM_CG_H
#define CUFOAM_CG_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamCG : public lduMatrix::solver {
    cuFoamCG(const cuFoamCG &) = delete;
    cuFoamCG &operator=(const cuFoamCG &) = delete;

public:
    TypeName("cuFoamCG");

    cuFoamCG(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamCG() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_CG_H
