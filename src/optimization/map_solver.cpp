#include "optimization/map_solver.h"

#include <memory>
#include <utility>
#include <vector>

#include "optimization/regularizer.h"

#include "glog/logging.h"

namespace super_resolution {

MapSolver::MapSolver(
    const ImageModel& image_model,
    const std::vector<ImageData>& low_res_images,
    const bool print_solver_output)
    : Solver(image_model, print_solver_output) {

  const int num_observations = low_res_images.size();
  CHECK_GT(num_observations, 0)
      << "Cannot super-resolve with 0 low-res images.";

  // Set number of channels, and verify that this is consistent among all of
  // the given low-res images.
  num_channels_ = low_res_images[0].GetNumChannels();
  for (int i = 1; i < low_res_images.size(); ++i) {
    CHECK_EQ(low_res_images[i].GetNumChannels(), num_channels_)
        << "Image channel counts do not match up.";
  }

  // Set the size of the HR images. There must be at least one image at
  // low_res_images[0], otherwise the above check will have failed.
  const int upsampling_scale = image_model_.GetDownsamplingScale();
  const cv::Size lr_image_size = low_res_images[0].GetImageSize();
  image_size_ = cv::Size(
      lr_image_size.width * upsampling_scale,
      lr_image_size.height * upsampling_scale);

  // Rescale the LR observations to the HR image size so they're useful for in
  // the objective function.
  observations_.reserve(num_observations);
  for (const ImageData& low_res_image : low_res_images) {
    ImageData observation = low_res_image;  // copy
    observation.ResizeImage(image_size_, INTERPOLATE_NEAREST);
    observations_.push_back(observation);
  }
}

void MapSolver::AddRegularizer(
    const Regularizer& regularizer,
    const double regularization_parameter) {

  regularizers_.push_back(
      std::make_pair(&regularizer, regularization_parameter));
}

}  // namespace super_resolution
