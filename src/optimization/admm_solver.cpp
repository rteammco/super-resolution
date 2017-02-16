#include "optimization/admm_solver.h"

#include <memory>

#include "optimization/objective_data_term.h"
#include "optimization/objective_function.h"

namespace super_resolution {

ImageData AdmmSolver::Solve(const ImageData& initial_estimate) {
  const int num_channels = GetNumChannels();
  const int num_data_points = GetNumDataPoints();

  ObjectiveFunction data_term_objective(num_data_points);
  std::shared_ptr<ObjectiveTerm> data_term(new ObjectiveDataTerm(
      image_model_, observations_, num_channels, GetImageSize()));
  data_term_objective.AddTerm(data_term);
  // ObjectiveEqualityConstraintTerm  TODO: implement this.

  ObjectiveFunction regularization_objective(num_data_points);
  // ObjectiveRegularizationTerm ?
  // ObjectiveEqualityConstraintTerm

  // TODO: verify that this is the correct approach and then implement it.
  // X = Z = initial_estimate;
  // while (not converged) {
  //   Z = solve(data_term_objective) <- X, Z
  //   X = solve(regularization_objective) <- X, Z
  // }

  // TODO: return the solved Image.
  return initial_estimate;
}

}  // namespace super_resolution
