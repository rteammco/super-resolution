#include "solvers/map_solver.h"

#include <iostream>
#include <vector>

#include "image/image_data.h"
#include "solvers/map_cost_function.h"
#include "solvers/map_cost_processor.h"

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
    // TODO: implement
    LOG(INFO) << "CALLBACK";
    return ceres::SOLVER_CONTINUE;
  }
};

ImageData MapSolver::Solve(const ImageData& initial_estimate) const {
  const int num_observations = low_res_images_.size();
  CHECK(num_observations > 0) << "Cannot solve with 0 low-res images.";

  const cv::Size hr_image_size = initial_estimate.GetImageSize();
  const int num_hr_pixels = initial_estimate.GetNumPixels();
  const MapCostProcessor map_cost_processor(
      low_res_images_, image_model_, hr_image_size);

  LOG(INFO) << "Created MapCostProc";

  ImageData estimated_image = initial_estimate;

  const int num_channels = low_res_images_[0].GetNumChannels();
  const int num_images = low_res_images_.size();

  LOG(INFO) << "Ready to build problem";

  ceres::Problem problem;
  double* data_ptr = estimated_image.GetMutableDataPointer(0, 0);
  // TODO: currently solves independently for each channel.
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    for (int image_index = 0; image_index < num_images; ++image_index) {
      ceres::CostFunction* cost_function = MapCostFunction::Create(
          image_index, channel_index, num_hr_pixels, map_cost_processor);
      problem.AddResidualBlock(
          cost_function,
          NULL,  // basic loss
          // TODO: pointer to the whole image, not one pixel! This will work if
          // index at 0 though...
          data_ptr);
          //estimated_image.GetMutableDataPointer(channel_index, 0));
    }
  }

  LOG(INFO) << "Problem done.";

  // TODO: handle regularization.

  // Set the solver options. TODO: figure out what these should be.
  ceres::Solver::Options options;
  // options.linear_solver_type = ceres::DENSE_SCHUR;
  // Always update parameters because we need to compute the new LR estimates.
  options.update_state_every_iteration = true;
  options.callbacks.push_back(new ApplyModelCallback());
  options.minimizer_progress_to_stdout = true;

  // Solve.
  LOG(INFO) << "About to solve.";
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << "SOLVED.";
  std::cout << summary.FullReport() << std::endl;

  for (int i = 0; i < num_hr_pixels; ++i) {
    LOG(INFO) << "Final pixel: " << data_ptr[i];
  }

  return estimated_image;
}

}  // namespace super_resolution
