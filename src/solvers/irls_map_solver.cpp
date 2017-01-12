#include "solvers/irls_map_solver.h"

#include <utility>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

ImageData IrlsMapSolver::Solve(const ImageData& initial_estimate) const {
  // TODO: implement.
  return initial_estimate;
}

std::pair<double, std::vector<double>>
IrlsMapSolver::ComputeDataTermAnalyticalDiff(
    const int image_index,
    const int channel_index,
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  // Degrade (and re-upsample) the HR estimate with the image model.
  ImageData degraded_hr_image(estimated_image_data, image_size_);
  image_model_.ApplyToImage(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size_, cv::INTER_NEAREST);

  const int num_pixels = image_size_.width * image_size_.height;

  // Compute the individual residuals by comparing pixel values. Sum them up
  // for the final residual sum.
  double residual_sum = 0;
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    const double residual =
        degraded_hr_image.GetPixelValue(0, i) -
        observations_.at(image_index).GetPixelValue(channel_index, i);
    residuals.push_back(residual);
    residual_sum += (residual * residual);
  }

  // Apply transpose operations to the residual image. This is used to compute
  // the gradient.
  ImageData residual_image(residuals.data(), image_size_);
  const int scale = image_model_.GetDownsamplingScale();
  residual_image.ResizeImage(cv::Size(
      image_size_.width / scale,
      image_size_.height / scale));
  image_model_.ApplyTransposeToImage(&residual_image, image_index);

  // Build the gradient vector.
  std::vector<double> gradient;
  gradient.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    // Only 1 channel (channel 0) in the residual image.
    const double partial_derivative = 2 * residual_image.GetPixelValue(0, i);
    gradient.push_back(partial_derivative);
  }

  return make_pair(residual_sum, gradient);
}

}  // namespace super_resolution
