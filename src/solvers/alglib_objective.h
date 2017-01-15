// The definition of the objective function when using ALGLIB.

#ifndef SRC_SOLVERS_IRLS_ALGLIB_OBJECTIVE_H_
#define SRC_SOLVERS_IRLS_ALGLIB_OBJECTIVE_H_

#include "alglib/src/stdafx.h"
#include "alglib/src/optimization.h"

namespace super_resolution {

// The objective function used by the ALGLIB solver to compute residuals. This
// version uses analyitical differentiation, meaning that the gradient is
// computed manually.
void AlglibObjectiveFunctionAnalyticalDiff(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradient,  // NOLINT
    void* irls_map_solver_ptr);

// The callback function for the ALGLIB solver. Called after every solver
// iteration, which updates the IRLS weights.
void AlglibSolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* irls_map_solver_ptr);

}  // namespace super_resolution

#endif  // SRC_SOLVERS_IRLS_ALGLIB_OBJECTIVE_H_
