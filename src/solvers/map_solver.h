// The MapSolver is an implementation framework for the maximum a posteriori
// formulation. This is a pure virtual class, since the MAP formulation can be
// solved by different strategies, such as IRLS (iteratively reweighted least
// squares). See subclasses for the specific solver strategy implementations.

#ifndef SRC_SOLVERS_MAP_SOLVER_H_
#define SRC_SOLVERS_MAP_SOLVER_H_

#include <memory>
#include <utility>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "solvers/regularizer.h"
#include "solvers/solver.h"

namespace super_resolution {

// TODO: remove this.
enum RegularizationMethod {
  TOTAL_VARIATION
};

struct MapSolverOptions {
  // TODO: fill in the options as needed.
  // Using temporary (old) options to avoid compile errors.
  double regularization_parameter = 0.0;
  RegularizationMethod regularization_method = TOTAL_VARIATION;
  bool use_numerical_differentiation = false;
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

  // Adds a regularizer term to the objective function.
  void AddRegularizer(  // TODO: virtual or not?
      std::unique_ptr<Regularizer> regularizer,
      const double regularization_parameter);

  // TODO: remove.
  virtual ImageData Solve(const ImageData& initial_estimate) const {
    return initial_estimate;
  }

 protected:
  const MapSolverOptions solver_options_;
  std::vector<std::pair<std::unique_ptr<Regularizer>, double>> regularizers_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_SOLVER_H_
