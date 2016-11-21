// The MapSolver is an implementation for the maximum a posteriori formulation.
// It solves the MAP formulation using iteratively reweighted least squares.

#ifndef SRC_SOLVERS_MAP_SOLVER_H_
#define SRC_SOLVERS_MAP_SOLVER_H_

#include <vector>

#include "image/image_data.h"
#include "solvers/solver.h"

namespace super_resolution {

// RegularizationMethods map to a Regularizer object that computes the
// regularization cost of the objective. In this formulation, the regularizer
// should ;
enum RegularizationMethod {
  TOTAL_VARIATION,
  BILATERAL_TOTAL_VARIATION  // TODO: not implemented
  // TODO: Add more regularizers as appropriate.
};

// Options for this solver.
struct MapSolverOptions {
  // The regularization parameter is a hyperparameter that scales the
  // regularization cost. If it is zero, the MAP formulation becomes maximum
  // likelihood, as the regularization term will have no effect.
  double regularization_parameter = 0.0;

  // Choice of RegularizationMethod for the regularization cost functor.
  RegularizationMethod regularization_method = TOTAL_VARIATION;
};

class MapSolver : public Solver {
 public:
  // Constructor is the same as Solver constructor but also takes the
  // MapSolverOptions struct.
  MapSolver(
      const MapSolverOptions& solver_options,
      const ImageModel& image_model,
      const std::vector<ImageData>& low_res_images,
      const bool print_solver_output = true) :
    Solver(image_model, low_res_images, print_solver_output),
    solver_options_(solver_options) {}

  // Solves the problem using the MAP (maximum a posteriori) formulation.
  virtual ImageData Solve(const ImageData& initial_estimate) const;

 private:
  const MapSolverOptions solver_options_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_SOLVER_H_
