// An iteratively reweighted least squares (IRLS) implementation of the MAP
// objective formulation.
#ifndef SRC_SOLVERS_IRLS_MAP_SOLVER_H_
#define SRC_SOLVERS_IRLS_MAP_SOLVER_H_

#include <utility>
#include <vector>

#include "image/image_data.h"
#include "solvers/map_solver.h"

namespace super_resolution {

class IrlsMapSolver : public MapSolver {
 public:
  using MapSolver::MapSolver;  // Inherit MapSolver constructor.

  // The IRLS MAP formulation solver implementation. Uses a least squares
  // solver library to do the actual optimization.
  virtual ImageData Solve(const ImageData& initial_estimate);

  // Computes the data fidelity term using analytical (manually computed)
  // differentiation. Returns a pair consisting of the sum of squared errors
  // (i.e. the residual) and the gradient vector values.
  std::pair<double, std::vector<double>> ComputeDataTermAnalyticalDiff(
      const int image_index,
      const int channel_index,
      const double* estimated_image_data) const;

  std::pair<double, std::vector<double>> ComputeRegularizationAnalyticalDiff(
      const double* estimated_image_data) const;

 private:
  // A vector containing the IRLS weights, one per parameter for which a
  // regularization residual is computed. These weights get reweighted after
  // every solver iteration to allow any regularizer norm under least squares.
  std::vector<double> irls_weights_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_IRLS_MAP_SOLVER_H_
