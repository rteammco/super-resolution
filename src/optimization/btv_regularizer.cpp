#include "optimization/btv_regularizer.h"

#include <cmath>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

BilateralTotalVariationRegularizer::BilateralTotalVariationRegularizer(
    const cv::Size& image_size,
    const int num_channels,
    const int scale_range,
    const double spatial_decay)
    : Regularizer(image_size, num_channels),
      scale_range_(scale_range),
      spatial_decay_(spatial_decay) {

  CHECK_GE(scale_range_, 1)
      << "Range must be at least 1 (1 pixel in each direction).";
  CHECK(0 < spatial_decay_ && spatial_decay_ < 1)
      << "Spatial decay must be between 0 and 1 (non-inclusive).";
}

std::vector<double> BilateralTotalVariationRegularizer::ApplyToImage(
    const double* image_data) const {

  CHECK_NOTNULL(image_data);

  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> residuals(num_pixels * num_channels_);
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
    const double* data_ptr = image_data + channel_index;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = channel_index + (row * image_size_.width + col);
        for (int i = 0; i <= scale_range_; ++i) {
          for (int j = 0; j <= scale_range_; ++j) {
            const int offset_row = row + i;
            const int offset_col = col + j;
            if (offset_row >= image_size_.height ||
                offset_col >= image_size_.width) {
              continue;
            }
            const int offset_index =
                channel_index + (offset_row * image_size_.width + offset_col);
            const double decay = std::pow(spatial_decay_, i + j);
            const double diff = image_data[index] - image_data[offset_index];
            residuals[index] += decay * std::abs(diff);
          }
        }
      }
    }
  }
  return residuals;
}

std::pair<std::vector<double>, std::vector<double>>
BilateralTotalVariationRegularizer::ApplyToImageWithDifferentiation(
    const double* image_data,
    const std::vector<double>& gradient_constants,
    const GradientComputationMethod& differentiation_method) const {

  CHECK_NOTNULL(image_data);

  // TODO: implement.
  const int num_pixels = image_size_.width * image_size_.height;
  const int num_parameters = num_pixels * num_channels_;
  std::vector<double> residual_values(num_parameters);
  std::vector<double> gradient(num_parameters);
  return std::make_pair(residual_values, gradient);
}

}  // namespace super_resolution
