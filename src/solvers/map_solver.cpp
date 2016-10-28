#include "solvers/map_solver.h"

#include <iostream>
#include <vector>

#include "image/image_data.h"
#include "solvers/map_cost_function.h"

#include "ceres/ceres.h"

namespace super_resolution {

ImageData MapSolver::Solve() const {
  CHECK(low_res_images_.size() > 0) << "Cannot solve with 0 images.";

  // Scale up the LR images so we can compare them to the HR estimate.
  // TODO: implement.
  // TODO: get the scale (as parameter? or from model...?).

  // 1. Compute an initial HR image estimate.
  // TODO: get an initial estimate.
  ImageData estimated_image;

  // 2. For each frame, apply the image model and naively scale it up to the
  // size of the HR image.
  // TODO: what type of interpolation to use for the scaling? cv::INTER_AREA?
  // TODO: implement this is a callback function/object.
  const int num_frames = low_res_images_.size();
  std::vector<ImageData> low_res_predictions;
  for (int i = 0; i < num_frames; ++i) {
    low_res_predictions.push_back(image_model_.ApplyModel(estimated_image, i));
    // TODO: rescale all of these LR predictions to match the HR size.
  }

  const int num_channels = low_res_images_[0].GetNumChannels();
  const int num_images = low_res_images_.size();
  const int num_pixels = low_res_images_[0].GetNumPixels();

  ceres::Problem problem;
  // TODO: currently solves independently for each channel.
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    for (int image_index = 0; image_index < num_images; ++image_index) {
      for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        ceres::CostFunction* cost_function = MapCostFunction::Create(
            &low_res_predictions, image_index, channel_index, pixel_index);
        problem.AddResidualBlock(
            cost_function,
            NULL,  // basic loss
            estimated_image.GetMutableDataPointer(channel_index, pixel_index));
      }
    }
  }

  // TODO: handle regularization.

  // Set the solver options. TODO: figure out what these should be.
  ceres::Solver::Options options;
  // options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  return estimated_image;
}

}  // namespace super_resolution
