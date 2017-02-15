// Defines functions that are used by the ALGLIB library for computing the
// objective cost and gradient values.

#ifndef SRC_OPTIMIZATION_ALGLIB_IRLS_OBJECTIVE_H_
#define SRC_OPTIMIZATION_ALGLIB_IRLS_OBJECTIVE_H_

#include "alglib/src/optimization.h"

namespace super_resolution {

// The objective function used by the ALGLIB solver to compute residuals. This
// version uses analyitical differentiation, meaning that the gradient is
// computed manually.
void AlglibObjectiveFunction(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradient,  // NOLINT
    void* objective_function_ptr);

// The same objective function as above, but does not compute the gradients.
// This is for numerical differentiation (test purposes only). This version of
// the objective function is very slow.
void AlglibObjectiveFunctionNumericalDiff(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    void* objective_function_ptr);

// The callback function for the ALGLIB solver. Called after every solver
// iteration, which updates the IRLS weights.
void AlglibSolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* objective_function_ptr);

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_ALGLIB_IRLS_OBJECTIVE_H_
