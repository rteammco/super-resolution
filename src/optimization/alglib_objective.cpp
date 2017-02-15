#include "optimization/alglib_objective.h"

#include <utility>
#include <vector>

#include "optimization/objective_function.h"

#include "alglib/src/optimization.h"

#include "glog/logging.h"

namespace super_resolution {

double RunCGSolverNumericalDiff(
    const MapSolverOptions& solver_options,
    const ObjectiveFunction& objective_function,
    alglib::real_1d_array* solver_data) {

  alglib::mincgstate solver_state;
  alglib::mincgreport solver_report;

  alglib::mincgcreatef(
      *solver_data,
      solver_options.numerical_differentiation_step,
      solver_state);

  alglib::mincgsetcond(
      solver_state,
      solver_options.gradient_norm_threshold,
      solver_options.cost_decrease_threshold,
      solver_options.parameter_variation_threshold,
      solver_options.max_num_solver_iterations);
  alglib::mincgsetxrep(solver_state, true);

  // Optimize with conjugate gradient.
  alglib::mincgoptimize(
      solver_state,
      AlglibObjectiveFunctionNumericalDiff,
      AlglibSolverIterationCallback,
      const_cast<void*>(reinterpret_cast<const void*>(&objective_function)));

  alglib::mincgresults(solver_state, *solver_data, solver_report);

  return solver_state.f;
}

double RunCGSolverAnalyticalDiff(
    const MapSolverOptions& solver_options,
    const ObjectiveFunction& objective_function,
    alglib::real_1d_array* solver_data) {

  alglib::mincgstate solver_state;
  alglib::mincgreport solver_report;

  alglib::mincgcreate(*solver_data, solver_state);

  alglib::mincgsetcond(
      solver_state,
      solver_options.gradient_norm_threshold,
      solver_options.cost_decrease_threshold,
      solver_options.parameter_variation_threshold,
      solver_options.max_num_solver_iterations);
  alglib::mincgsetxrep(solver_state, true);

  // Optimize with conjugate gradient.
  alglib::mincgoptimize(
      solver_state,
      AlglibObjectiveFunction,
      AlglibSolverIterationCallback,
      const_cast<void*>(reinterpret_cast<const void*>(&objective_function)));

  alglib::mincgresults(solver_state, *solver_data, solver_report);

  return solver_state.f;
}

void AlglibObjectiveFunction(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradient,  // NOLINT
    void* objective_function_ptr) {

  const ObjectiveFunction* objective_function =
      reinterpret_cast<ObjectiveFunction*>(objective_function_ptr);
  residual_sum = objective_function->ComputeAllTerms(
      estimated_data.getcontent(), gradient.getcontent());
}

void AlglibObjectiveFunctionNumericalDiff(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    void* objective_function_ptr) {

  const ObjectiveFunction* objective_function =
      reinterpret_cast<ObjectiveFunction*>(objective_function_ptr);
  residual_sum = objective_function->ComputeAllTerms(
      estimated_data.getcontent());
}

void AlglibSolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* objective_function_ptr) {

  LOG(INFO) << residual_sum;
// TODO: fix!
//  IrlsMapSolver* irls_map_solver =
//      reinterpret_cast<IrlsMapSolver*>(irls_map_solver_ptr);
//  irls_map_solver->NotifyIterationComplete(residual_sum);
}

}  // namespace super_resolution
