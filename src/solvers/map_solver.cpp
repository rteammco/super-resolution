#include "solvers/map_solver.h"

#include <memory>
#include <utility>

#include "solvers/regularizer.h"

namespace super_resolution {

void MapSolver::AddRegularizer(
    std::unique_ptr<Regularizer> regularizer,
    const double regularization_parameter) {

  regularizers_.push_back(
      std::make_pair(std::move(regularizer), regularization_parameter));
}

}  // namespace super_resolution
