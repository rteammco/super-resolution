// An iteratively reweighted least squares (IRLS) implementation of the MAP
// objective formulation.
#ifndef SRC_OPTIMIZATION_IRLS_MAP_SOLVER_H_
#define SRC_OPTIMIZATION_IRLS_MAP_SOLVER_H_

#include <utility>
#include <vector>

#include "image/image_data.h"
#include "optimization/map_solver.h"

namespace super_resolution {

struct IrlsMapSolverOptions : public MapSolverOptions {
  IrlsMapSolverOptions() {}  // Required for making a const instance.

  // Augments the adjustment to also include the irls cost difference.
  virtual void AdjustThresholdsAdaptively(
      const int num_parameters, const double regularization_parameter_sum) {
    const double threshold_scale =
        num_parameters * regularization_parameter_sum;
    if (threshold_scale < 1.0) {
      return;  // Only scale up if needed, not down.
    }
    MapSolverOptions::AdjustThresholdsAdaptively(
        num_parameters, regularization_parameter_sum);
    irls_cost_difference_threshold *= threshold_scale;
  }

  // Maximum number of outer loop iterations. Each outer loop runs Conjugate
  // Gradient which has its own max number of iterations
  // (max_num_solver_iterations).
  int max_num_irls_iterations = 20;

  // The cost difference threshold for convergence of the IRLS algorithm. If
  // the change in cost from one outer loop iteration to the next is below this
  // threshold, the solver will stop.
  //
  // The stopping criteria for the inner loop (conjugate gradient) is defined
  // independently in MapSolverOptions.
  double irls_cost_difference_threshold = 1.0e-5;
};

class IrlsMapSolver : public MapSolver {
 public:
  IrlsMapSolver(
      const IrlsMapSolverOptions& solver_options,
      const ImageModel& image_model,
      const std::vector<ImageData>& low_res_images,
      const bool print_solver_output = true);

  // The IRLS MAP formulation solver implementation. Uses a least squares
  // solver library to do the actual optimization.
  virtual ImageData Solve(const ImageData& initial_estimate);

 private:
  // Passed in through the constructor.
  const IrlsMapSolverOptions solver_options_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_IRLS_MAP_SOLVER_H_
