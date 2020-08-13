#ifndef CUFOAM_MRS_H
#define CUFOAM_MRS_H

#include "lduMatrix.H"

namespace Foam {

class cuFoamMRS : public lduMatrix::solver {
    cuFoamMRS(const cuFoamMRS &) = delete;
    cuFoamMRS &operator=(const cuFoamMRS &) = delete;

public:
    TypeName("cuFoamMRS");

    cuFoamMRS(
        const word &fieldName,
        const lduMatrix &matrix,
        const FieldField<Field, scalar> &coupleBouCoeffs,
        const FieldField<Field, scalar> &coupleIntCoeffs,
        const lduInterfaceFieldPtrsList &interfaces,
        const dictionary &solverControls
    );

    virtual ~cuFoamMRS() {}

    virtual solverPerformance solve(
        scalarField &psi,
        const scalarField &source,
        const direction cmpt = 0
    ) const;
};

}  // namespace Foam

#endif // ~CUFOAM_MRS_H
