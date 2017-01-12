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
  ImageData Solve(const ImageData& initial_estimate) const;

  // Computes the data fidelity term using analytical (manually computed)
  // differentiation. Returns a pair consisting of the sum of squared errors
  // (i.e. the residual) and the gradient vector values.
  std::pair<double, std::vector<double>> ComputeDataTermAnalyticalDiff(
      const int image_index,
      const int channel_index,
      const double* estimated_data) const;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_IRLS_MAP_SOLVER_H_
