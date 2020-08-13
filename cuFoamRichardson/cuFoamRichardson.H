#ifndef CUFOAM_RICHARDSON_H
#define CUFOAM_RICHARDSON_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamRichardson : public lduMatrix::solver {
    cuFoamRichardson(const cuFoamRichardson &) = delete;
    cuFoamRichardson &operator=(const cuFoamRichardson &) = delete;

public:
    TypeName("cuFoamRichardson");

    cuFoamRichardson(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamRichardson() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_RICHARDSON_H
