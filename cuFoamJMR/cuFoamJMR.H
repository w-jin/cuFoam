#ifndef CUFOAM_JMR_H
#define CUFOAM_JMR_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamJMR : public lduMatrix::solver {
    cuFoamJMR(const cuFoamJMR &) = delete;
    cuFoamJMR &operator=(const cuFoamJMR &) = delete;

public:
    TypeName("cuFoamJMR");

    cuFoamJMR(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamJMR() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_JMR_H
