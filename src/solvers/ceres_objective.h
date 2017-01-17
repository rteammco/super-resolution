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
//class MapDataCostFunction : public ceres::FirstOrderFunction {
class MapDataCostFunction : public ceres::CostFunction {
 public:
  MapDataCostFunction(const IrlsMapSolver* irls_map_solver)
      : irls_map_solver_(irls_map_solver),
        num_pixels_(irls_map_solver->GetNumPixels()) {

    mutable_parameter_block_sizes()->push_back(num_pixels_);
    set_num_residuals(1);
  }

  //virtual int NumParameters() const {
  //  return num_pixels_;
  //}

  virtual bool Evaluate(
  //    const double* parameters, double* residuals, double* jacobian) const {
      double const* const* parameters,
      double* residuals,
      double** jacobian) const {

    // Get the pararmaeters.
    const double* estimated_image_data = parameters[0];
    double* cost = residuals;
    double* gradient = nullptr;
    if (jacobian != NULL) {
      gradient = jacobian[0];
    }

    // Zero-out the gradient if the gradient vector is given.
    if (gradient != nullptr) {
      for (int i = 0; i < num_pixels_; ++i) {
        gradient[i] = 0;
      }
    }
    cost[0] = 0.0;

    // Loop through all the LR images to compute the cost and gradient.
    const int num_images = irls_map_solver_->GetNumImages();
    for (int image_index = 0; image_index < num_images; ++image_index) {
      const int num_images = irls_map_solver_->GetNumImages();
      const auto& residual_sum_and_gradient =
          irls_map_solver_->ComputeDataTermAnalyticalDiff(
              image_index, 0, estimated_image_data);  // TODO: channel!?
      if (gradient != nullptr) {
        for (int i = 0; i < num_pixels_; ++i) {
          gradient[i] += residual_sum_and_gradient.second[i];
        }
      }
      cost[0] += residual_sum_and_gradient.first;
    }

    LOG(INFO) << "COST";
    return true;
  }

 private:
  const IrlsMapSolver* irls_map_solver_;
  const int num_pixels_;
};

// TODO: this should only update W. Fix.
class MapIterationCallback : public ceres::IterationCallback {
 public:
  MapIterationCallback(
      double* hr_image_estimate,
      IrlsMapSolver* irls_map_solver)
      : hr_image_estimate_(hr_image_estimate),
        irls_map_solver_(irls_map_solver) {}

  // Called after each iteration.
  ceres::CallbackReturnType operator() (
      const ceres::IterationSummary& summary) {
    // TODO: implement computing W matrix.
    // TODO: remove logging here.
    irls_map_solver_->UpdateIrlsWeights(hr_image_estimate_);
    LOG(INFO) << "CALLBACK:\n"
        << hr_image_estimate_[0] << ", "
        << hr_image_estimate_[1] << ", "
        << hr_image_estimate_[2] << ", "
        << hr_image_estimate_[3] << ", "
        << hr_image_estimate_[4] << ", "
        << hr_image_estimate_[5] << ", "
        << hr_image_estimate_[6] << ", "
        << hr_image_estimate_[7] << ", ";
    return ceres::SOLVER_CONTINUE;
  }

 private:
  double* hr_image_estimate_;
  IrlsMapSolver* irls_map_solver_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_CERES_OBJECTIVE_H_
