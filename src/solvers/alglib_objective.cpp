#include "solvers/alglib_objective.h"

#include <utility>
#include <vector>

#include "solvers/irls_map_solver.h"

#include "alglib/src/optimization.h"

#include "glog/logging.h"

namespace super_resolution {

void AlglibObjectiveFunctionAnalyticalDiff(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradient,  // NOLINT
    void* irls_map_solver_ptr) {

  IrlsMapSolver* irls_map_solver =
      reinterpret_cast<IrlsMapSolver*>(irls_map_solver_ptr);

  // Zero out the residual sum and the gradient vector.
  residual_sum = 0;
  const int num_pixels = irls_map_solver->GetNumPixels();
  for (int i = 0; i < num_pixels; ++i) {
    gradient[i] = 0;
  }

  LOG(INFO) << "ITER START: " << estimated_data[0];

  // Compute data term residuals and gradient.
  const int num_images = irls_map_solver->GetNumImages();
  for (int image_index = 0; image_index < num_images; ++image_index) {
    const std::pair<double, std::vector<double>> residual_sum_and_gradient =
        irls_map_solver->ComputeDataTermAnalyticalDiff(
            image_index, 0, estimated_data.getcontent());  // TODO: channel!?
    residual_sum += residual_sum_and_gradient.first;
    for (int i = 0; i < num_pixels; ++i) {
      gradient[i] += residual_sum_and_gradient.second[i];
    }
  }

  // Compute regularization residuals and gradient.
  const std::pair<double, std::vector<double>> residual_sum_and_gradient =
      //irls_map_solver->ComputeRegularizationAnalyticalDiff(  // TODO!
      irls_map_solver->ComputeRegularizationAutomaticDiff(
          estimated_data.getcontent());
  residual_sum += residual_sum_and_gradient.first;
  for (int i = 0; i < num_pixels; ++i) {
    gradient[i] += residual_sum_and_gradient.second[i];
  }

  // TODO: also add an optional numerical differentiation objective as before.
}

void AlglibSolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* irls_map_solver_ptr) {

  IrlsMapSolver* irls_map_solver =
      reinterpret_cast<IrlsMapSolver*>(irls_map_solver_ptr);
  irls_map_solver->UpdateIrlsWeights(estimated_data.getcontent());
  LOG(INFO) << "Callback: residual sum = " << residual_sum;
}

}  // namespace super_resolution
