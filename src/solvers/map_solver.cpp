#include "solvers/map_solver.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "image/image_data.h"
#include "solvers/irls_cost_processor.h"
#include "solvers/map_data_cost_functor.h"
#include "solvers/map_regularization_cost_functor.h"
#include "solvers/regularizer.h"
#include "solvers/tv_regularizer.h"

#include "opencv2/core/core.hpp"

#include "ceres/ceres.h"

#include "glog/logging.h"

namespace super_resolution {

constexpr double kMinIrlsWeight = 0.0001;  // Used to avoid division by 0.

// The IRLS callback updates the regularization weights of after each solver
// iteration.
class IrlsCallback : public ceres::IterationCallback {
 public:
  IrlsCallback(
      const ImageData& estimated_image,
      const IrlsCostProcessor& irls_cost_processor,
      std::vector<double>* irls_weights)
    : estimated_image_(estimated_image),
      irls_cost_processor_(irls_cost_processor),
      irls_weights_(irls_weights) {}

  // Called after each iteration.
  ceres::CallbackReturnType operator() (
      const ceres::IterationSummary& summary) {
    // TODO: make a function exist where you can get a non-mutable data pointer
    // just to be safe.
    // TODO: Channel = 0! Set to appropriate channel!!
    const std::vector<double> regularization_residuals =
        irls_cost_processor_.ComputeRegularizationResiduals(
            estimated_image_.GetMutableDataPointer(0));  // TODO: channel 0!
    CHECK_EQ(regularization_residuals.size(), irls_weights_->size())
        << "Number of residuals does not match number of weights.";
    for (int i = 0; i < regularization_residuals.size(); ++i) {
      (*irls_weights_)[i] =
          1.0 / std::max(kMinIrlsWeight, regularization_residuals[i]);
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  // The estimated image, updated by the solver after every solver iteration.
  // The data here is updated every time before the callback is called.
  const ImageData& estimated_image_;

  // The IrlsCostProcessor is used to compute the regularization residuals of
  // the estimated image after the iteration to use for updating the weights.
  const IrlsCostProcessor& irls_cost_processor_;

  // The IRLS weights that are updated during the callback using the
  // regularization residuals on the estimated image.
  std::vector<double>* irls_weights_;
};

ImageData MapSolver::Solve(const ImageData& initial_estimate) const {
  const int num_observations = low_res_images_.size();
  CHECK(num_observations > 0) << "Cannot solve with 0 low-res images.";

  const cv::Size hr_image_size = initial_estimate.GetImageSize();
  const int num_hr_pixels = initial_estimate.GetNumPixels();
  std::unique_ptr<Regularizer> regularizer(
      new TotalVariationRegularizer(hr_image_size));

  // Initialize all IRLS weights to 1.
  std::vector<double> irls_weights(num_hr_pixels);
  std::fill(irls_weights.begin(), irls_weights.end(), 1);

  const IrlsCostProcessor irls_cost_processor(
      low_res_images_,
      image_model_,
      hr_image_size,
      std::move(regularizer),
      0.0,  // TODO: (lambda) this should be passed in as a user option.
      &irls_weights);

  ImageData estimated_image = initial_estimate;

  const int num_channels = low_res_images_[0].GetNumChannels();
  const int num_images = low_res_images_.size();

  ceres::Problem problem;
  // TODO: currently solves independently for each channel.
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    // Set up the data fidelity cost function per each image.
    for (int image_index = 0; image_index < num_images; ++image_index) {
      ceres::CostFunction* data_cost_function = MapDataCostFunctor::Create(
          image_index, channel_index, num_hr_pixels, irls_cost_processor);
      problem.AddResidualBlock(
          data_cost_function,
          NULL,  // basic loss TODO: update?
          estimated_image.GetMutableDataPointer(channel_index));
    }
    // Set up the regularization cost function for the channel.
    ceres::CostFunction* regularization_cost_function =
        MapRegularizationCostFunctor::Create(
            num_hr_pixels, irls_cost_processor);
    problem.AddResidualBlock(
        regularization_cost_function,
        NULL,  // basic loss TODO: update?
        estimated_image.GetMutableDataPointer(channel_index));
  }

  // Set the solver options. TODO: figure out what these should be.
  ceres::Solver::Options options;
  // options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.num_threads = 4;
  // options.num_linear_solver_threads = 4;
  // Always update parameters because we need to compute the new LR estimates.
  options.update_state_every_iteration = true;
  options.callbacks.push_back(
      new IrlsCallback(estimated_image, irls_cost_processor, &irls_weights));
  options.minimizer_progress_to_stdout = print_solver_output_;

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (print_solver_output_) {
    std::cout << summary.FullReport() << std::endl;
  }

  return estimated_image;
}

}  // namespace super_resolution
