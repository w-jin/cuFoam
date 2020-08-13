#ifndef CUFOAM_MR_H
#define CUFOAM_MR_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamMR : public lduMatrix::solver {
    cuFoamMR(const cuFoamMR &) = delete;
    cuFoamMR &operator=(const cuFoamMR &) = delete;

public:
    TypeName("cuFoamMR");

    cuFoamMR(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamMR() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_MR_H
