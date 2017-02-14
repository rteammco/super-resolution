// An iteratively reweighted least squares (IRLS) implementation of the MAP
// objective formulation.
#ifndef SRC_OPTIMIZATION_IRLS_MAP_SOLVER_H_
#define SRC_OPTIMIZATION_IRLS_MAP_SOLVER_H_

#include <utility>
#include <vector>

#include "image/image_data.h"
#include "optimization/map_solver.h"

namespace super_resolution {

class IrlsMapSolver : public MapSolver {
 public:
  IrlsMapSolver(
      const MapSolverOptions& solver_options,
      const ImageModel& image_model,
      const std::vector<ImageData>& low_res_images,
      const bool print_solver_output = true);

  // The IRLS MAP formulation solver implementation. Uses a least squares
  // solver library to do the actual optimization.
  virtual ImageData Solve(const ImageData& initial_estimate);

  // Computes the data fidelity term using analytical (manually computed)
  // differentiation. Returns the sum of squared errors (i.e. the residual).
  // The gradient is added to the given gradient array. If the gradient is
  // null, it will not be computed.
  double ComputeDataTerm(
      const int image_index,
      const double* estimated_image_data,
      double* gradient = nullptr) const;

  // Computes the regularization term and its gradient. If the gradient is
  // null, it will not be computed.
  double ComputeRegularization(
    const double* estimated_image_data, double* gradient = nullptr) const;

  // Notifies the Solver
  void NotifyIterationComplete(const double total_residual_sum);

 private:
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
