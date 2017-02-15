// Defines functions that are used by the ALGLIB library for computing the
// objective cost and gradient values for an objective function solved with
// conjugate gradient.

#ifndef SRC_OPTIMIZATION_ALGLIB_OBJECTIVE_H_
#define SRC_OPTIMIZATION_ALGLIB_OBJECTIVE_H_

#include "optimization/map_solver.h"
#include "optimization/objective_function.h"

#include "alglib/src/optimization.h"

namespace super_resolution {

// Sets up and runs the conjugate gradient solver (using ALGLIB's
// implementation) in numerical differentiation mode. The given solver_data
// will be modified and should be initialized beforehand with the initial data
// estimate.
//
// Returns the final objective cost value.
double RunCGSolverNumericalDiff(
    const MapSolverOptions& solver_options,
    const ObjectiveFunction& objective_function,
    alglib::real_1d_array* solver_data);

// Sets up and runs the conjugate gradient solver in analytical differentiation
// mode, similarly to RunCGSolverNumericalDiff().
double RunCGSolverAnalyticalDiff(
    const MapSolverOptions& solver_options,
    const ObjectiveFunction& objective_function,
    alglib::real_1d_array* solver_data);

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
// iteration.
void AlglibSolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* objective_function_ptr);

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_ALGLIB_OBJECTIVE_H_
