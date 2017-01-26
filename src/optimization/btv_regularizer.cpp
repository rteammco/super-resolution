#include "optimization/btv_regularizer.h"

#include <cmath>
#include <utility>
#include <vector>

#include "optimization/regularizer.h"

#include "opencv2/core/core.hpp"

#include "FADBAD++/fadiff.h"

#include "glog/logging.h"

using fadbad::F;  // FADBAD++ forward derivative template.

namespace super_resolution {

// Returns the bilateral total variation value for the pixel at the given row
// and col. Returned value will be 0 for invalid row and col values.
template<typename T>
T GetBilateralTotalVariation(
    const T* image_data,
    const cv::Size& image_size,
    const int row,
    const int col,
    const int scale_range,
    const double spatial_decay) {

  T total_variation = T(0);
  const int index = row * image_size.width + col;
  for (int i = 0; i <= scale_range; ++i) {
    for (int j = 0; j <= scale_range; ++j) {
      const int offset_row = row + i;
      const int offset_col = col + j;
      if (offset_row >= image_size.height || offset_col >= image_size.width) {
        continue;
      }
      const int offset_index = offset_row * image_size.width + offset_col;
      const T decay = T(std::pow(spatial_decay, i + j));
      T absdiff = image_data[index] - image_data[offset_index];
      if (absdiff < T(0)) {
        absdiff *= T(-1);
      }
      total_variation += decay * absdiff;
    }
  }
  return total_variation;
}

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
        residuals[index] = GetBilateralTotalVariation<double>(
            data_ptr, image_size_, row, col, scale_range_, spatial_decay_);
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

  // Initialize the derivatives of each parameter with respect to itself.
  const int num_pixels = image_size_.width * image_size_.height;
  const int num_parameters = num_pixels * num_channels_;
  std::vector<F<double>> parameters(
      image_data, image_data + num_parameters);
  for (int i = 0; i < num_parameters; ++i) {
    parameters[i].diff(i, num_parameters);
  }

  // TODO: copy/pasted >.>
  std::vector<double> residual_values(num_parameters);
  std::vector<F<double>> residuals(num_parameters);
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
    const F<double>* data_ptr = parameters.data() + channel_index;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = channel_index + (row * image_size_.width + col);
        residuals[index] = GetBilateralTotalVariation<F<double>>(
            data_ptr, image_size_, row, col, scale_range_, spatial_decay_);
        residual_values[index] = residuals[index].x();
      }
    }
  }

  // Compute the gradient vector. Only consider gradients of pixels within
  // range of each other to avoid O(n^2) comparison.
  std::vector<double> gradient(num_parameters);
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = channel_index + (row * image_size_.width + col);
        for (int i = 0; i <= scale_range_; ++i) {
          for (int j = 0; j <= scale_range_; ++j) {
            const int offset_row = row - i;
            const int offset_col = col - j;
            if (offset_row >= 0 && offset_col >= 0) {
              const int offset_index =
                  channel_index + (offset_row * image_size_.width + offset_col);
              const double dodi = residuals[offset_index].d(index);
              if (!isnan(dodi)) {
                gradient[index] +=
                    2 *
                    gradient_constants[offset_index] *
                    residual_values[offset_index] *
                    dodi;
              }
            }
          }
        }
      }
    }
  }

  return std::make_pair(residual_values, gradient);
}

}  // namespace super_resolution
