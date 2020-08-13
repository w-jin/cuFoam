#ifndef CUFOAM_BICG_H
#define CUFOAM_BICG_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamBiCG : public lduMatrix::solver {
    cuFoamBiCG(const cuFoamBiCG &) = delete;
    cuFoamBiCG &operator=(const cuFoamBiCG &) = delete;

public:
    TypeName("cuFoamBiCG");

    cuFoamBiCG(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamBiCG() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_BICG_H
