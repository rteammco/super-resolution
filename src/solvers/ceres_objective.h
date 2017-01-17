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
class MapDataCostFunction : public ceres::FirstOrderFunction {
 public:
  MapDataCostFunction(
    const IrlsMapSolver* irls_map_solver, const int image_index)
    : irls_map_solver_(irls_map_solver),
      image_index_(image_index),
      num_pixels_(irls_map_solver->GetNumPixels()) {

    // TODO: PUT BACK ANd ALSO HERE
    //mutable_parameter_block_sizes()->push_back(num_pixels_);
    //set_num_residuals(1);
  }

  virtual int NumParameters() const {
    return num_pixels_;
  }

  virtual bool Evaluate(
      const double* parameters,
      //double const* const* parameters,
      double* residuals,
      double* jacobians) const {
      //double** jacobians) const {

    // Zero-out the gradients if the jacobians matrix is given.
    double* gradient = nullptr;
    if (jacobians != NULL) {
      gradient = jacobians;//[0];
      for (int i = 0; i < num_pixels_; ++i) {
        gradient[i] = 0;
      }
    }
    residuals[0] = 0.0;

    for (int image_index = 0; image_index < irls_map_solver_->GetNumImages(); ++image_index) {
    const double* estimated_image_data = parameters;//[0];
    const int num_images = irls_map_solver_->GetNumImages();
    const auto& residual_sum_and_gradient =
        irls_map_solver_->ComputeDataTermAnalyticalDiff(
            image_index, 0, estimated_image_data);  // TODO: channel!?
    if (gradient != nullptr) {
      for (int i = 0; i < num_pixels_; ++i) {
        gradient[i] += residual_sum_and_gradient.second[i];
      }
    }
    residuals[0] += sqrt(residual_sum_and_gradient.first);
    }

    return true;
  }

 private:
  const IrlsMapSolver* irls_map_solver_;
  const int image_index_;
  const int num_pixels_;
};

// TODO: this should only update W. Fix.
class MapIterationCallback : public ceres::IterationCallback {
 public:
  // Called after each iteration.
  ceres::CallbackReturnType operator() (
      const ceres::IterationSummary& summary) {
    // TODO: implement computing W matrix.
    // TODO: remove logging here.
    LOG(INFO) << "CALLBACK";
    return ceres::SOLVER_CONTINUE;
  }
};


}  // namespace super_resolution

#endif  // SRC_SOLVERS_CERES_OBJECTIVE_H_
