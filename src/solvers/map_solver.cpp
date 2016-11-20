#include "solvers/map_solver.h"

#include <iostream>
#include <memory>
#include <vector>

#include "image/image_data.h"
#include "solvers/map_data_cost_functor.h"
#include "solvers/map_cost_processor.h"
#include "solvers/regularizer.h"
#include "solvers/tv_regularizer.h"

#include "opencv2/core/core.hpp"

#include "ceres/ceres.h"

#include "glog/logging.h"

namespace super_resolution {

// TODO: this should only update W. Fix.
class ApplyModelCallback : public ceres::IterationCallback {
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

ImageData MapSolver::Solve(const ImageData& initial_estimate) const {
  const int num_observations = low_res_images_.size();
  CHECK(num_observations > 0) << "Cannot solve with 0 low-res images.";

  const cv::Size hr_image_size = initial_estimate.GetImageSize();
  const int num_hr_pixels = initial_estimate.GetNumPixels();
  std::unique_ptr<Regularizer> regularizer(
      new TotalVariationRegularizer(0.1, hr_image_size));
  const MapCostProcessor map_cost_processor(
      low_res_images_, image_model_, hr_image_size, std::move(regularizer));

  ImageData estimated_image = initial_estimate;

  const int num_channels = low_res_images_[0].GetNumChannels();
  const int num_images = low_res_images_.size();

  ceres::Problem problem;
  // TODO: currently solves independently for each channel.
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    for (int image_index = 0; image_index < num_images; ++image_index) {
      ceres::CostFunction* cost_function = MapDataCostFunctor::Create(
          image_index, channel_index, num_hr_pixels, map_cost_processor);
      problem.AddResidualBlock(
          cost_function,
          NULL,  // basic loss
          estimated_image.GetMutableDataPointer(channel_index));
    }
  }

  // TODO: handle regularization.

  // Set the solver options. TODO: figure out what these should be.
  ceres::Solver::Options options;
  // options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.num_threads = 4;
  // options.num_linear_solver_threads = 4;
  // Always update parameters because we need to compute the new LR estimates.
  options.update_state_every_iteration = true;
  options.callbacks.push_back(new ApplyModelCallback());
  options.minimizer_progress_to_stdout = true;

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  return estimated_image;
}

}  // namespace super_resolution
