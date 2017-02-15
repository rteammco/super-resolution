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

  // Maximum number of outer loop iterations. Each outer loop runs Conjugate
  // Gradient which has its own max number of iterations
  // (max_num_solver_iterations).
  int max_num_irls_iterations = 50;

  // The cost difference threshold for convergence of the IRLS algorithm. If
  // the change in cost from one outer loop iteration to the next is below this
  // threshold, the solver will stop.
  //
  // The stopping criteria for the inner loop (conjugate gradient) is defined
  // independently in MapSolverOptions.
  double irls_cost_difference_threshold = 1.0e-6;
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

  // Notifies the Solver
  void NotifyIterationComplete(const double total_residual_sum);

 private:
  // Passed in through the constructor.
  const IrlsMapSolverOptions solver_options_;

  // A vector containing the IRLS weights, one per parameter of the solver
  // system. These weights get reweighted when the solver finishes and the
  // system solves is run again. The effect of reweighting is to allow solving
  // a 1-norm (or arbitrary p-norm) regularizer with least squares.
  //
  // TODO: one set of weights per regularizer.
  std::vector<double> irls_weights_;

  // The residual sum from the last iteration of the solver. This is updated by
  // calling NotifyIterationComplete with the correct residual sum value for
  // that solver iteration.
  double last_iteration_residual_sum_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_IRLS_MAP_SOLVER_H_
