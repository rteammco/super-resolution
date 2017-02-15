#include "optimization/alglib_irls_objective.h"

#include <utility>
#include <vector>

#include "optimization/objective_function.h"

#include "alglib/src/optimization.h"

#include "glog/logging.h"

namespace super_resolution {

void AlglibObjectiveFunction(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradient,  // NOLINT
    void* objective_function_ptr) {

  const ObjectiveFunction* objective_function =
      reinterpret_cast<ObjectiveFunction*>(objective_function_ptr);
  residual_sum = objective_function->ComputeAllTerms(
      estimated_data.getcontent(), gradient.getcontent());
//  // Need to zero out the residual sum and the gradient vector.
//  const int num_data_points = irls_map_solver->GetNumDataPoints();
//  for (int i = 0; i < num_data_points; ++i) {
//    gradient[i] = 0;
//  }
//
//  // Compute data term residuals and gradient.
//  const int num_images = irls_map_solver->GetNumImages();
//  for (int image_index = 0; image_index < num_images; ++image_index) {
//    residual_sum += irls_map_solver->ComputeDataTerm(
//        image_index, estimated_data.getcontent(), gradient.getcontent());
//  }
//
//  // Compute regularization residuals and gradient.
//  residual_sum += irls_map_solver->ComputeRegularization(
//      estimated_data.getcontent(), gradient.getcontent());
}

void AlglibObjectiveFunctionNumericalDiff(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    void* objective_function_ptr) {

  const ObjectiveFunction* objective_function =
      reinterpret_cast<ObjectiveFunction*>(objective_function_ptr);
  residual_sum = objective_function->ComputeAllTerms(
      estimated_data.getcontent());
//  IrlsMapSolver* irls_map_solver =
//      reinterpret_cast<IrlsMapSolver*>(irls_map_solver_ptr);
//
//  // Compute data term residuals.
//  residual_sum = 0;
//  const int num_images = irls_map_solver->GetNumImages();
//  for (int image_index = 0; image_index < num_images; ++image_index) {
//    residual_sum += irls_map_solver->ComputeDataTerm(
//        image_index, estimated_data.getcontent());
//  }
//
//  // Compute regularization residuals.
//  residual_sum += irls_map_solver->ComputeRegularization(
//      estimated_data.getcontent());
}

void AlglibSolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* irls_map_solver_ptr) {

  LOG(INFO) << residual_sum;
// TODO
//  IrlsMapSolver* irls_map_solver =
//      reinterpret_cast<IrlsMapSolver*>(irls_map_solver_ptr);
//  irls_map_solver->NotifyIterationComplete(residual_sum);
}

}  // namespace super_resolution
