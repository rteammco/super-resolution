// The super-resolution cost function (defined for Ceres) for the maximum a
// posteriori formulation data term. A different cost function is used for the
// regularization term.

#ifndef SRC_SOLVERS_CERES_OBJECTIVE_H_
#define SRC_SOLVERS_CERES_OBJECTIVE_H_

#include <vector>

#include "solvers/irls_map_solver.h"

#include "ceres/ceres.h"

namespace super_resolution {

// TODO: comments.
class MapDataCostFunction : public ceres::CostFunction {
  MapDataCostFunction(
      const IrlsMapSolver& irls_map_solver,
      const int image_index,
      const int channel_index)
    : irls_map_solver_(irls_map_solver),
      image_index_(image_index),
      channel_index_(channel_index) {

    mutable_parameter_block_sizes()->push_back(irls_map_solver.GetNumPixels());
    set_num_residuals(1);
  }

  virtual bool Evaluate(
      double const* const* parameters,
      double* residuals,
      double** jacobians) const {

    const double* estimated_image_data = parameters[0];
    double* gradient = jacobians[0];
    const std::pair<double, std::vector<double>> residual_sum_and_gradient =
        irls_map_solver_.ComputeDataTermAnalyticalDiff(
            image_index_, channel_index_, estimated_image_data);
    const double residual_sum = residual_sum_and_gradient.first;
    const int num_pixels = irls_map_solver_.GetNumPixels();
    for (int i = 0; i < num_pixels; ++i) {
      gradient[i] = residual_sum_and_gradient.second[i];
    }

    return true;
  }

 private:
  const IrlsMapSolver& irls_map_solver_;
  const int image_index_;
  const int channel_index_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_CERES_OBJECTIVE_H_
