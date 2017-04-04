#include "optimization/btv_regularizer.h"

#include <cmath>
#include <utility>
#include <vector>

#include "optimization/regularizer.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace {

// Returns the bilateral total variation value for the pixel at the given row
// and col. Returned value will be 0 for invalid row and col values.
double GetBilateralTotalVariation(
    const double* image_data,
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col,
    const int scale_range,
    const double spatial_decay) {

  double total_variation = 0.0;
  const int index = util::GetPixelIndex(image_size, channel, row, col);
  for (int i = 0; i <= scale_range; ++i) {
    for (int j = 0; j <= scale_range; ++j) {
      const int offset_row = row + i;
      const int offset_col = col + j;
      if (offset_row >= image_size.height || offset_col >= image_size.width) {
        continue;
      }
      const int offset_index =
          util::GetPixelIndex(image_size, channel, offset_row, offset_col);
      const double decay = std::pow(spatial_decay, i + j);
      const double absdiff =
          std::abs(image_data[index] - image_data[offset_index]);
      total_variation += decay * absdiff;
    }
  }
  return total_variation;
}

}  // namespace

BilateralTotalVariationRegularizer::BilateralTotalVariationRegularizer(
    const cv::Size& image_size,
    const int scale_range,
    const double spatial_decay)
    : Regularizer(image_size),
      scale_range_(scale_range),
      spatial_decay_(spatial_decay) {

  CHECK_GE(scale_range_, 1)
      << "Range must be at least 1 (1 pixel in each direction).";
  CHECK(0 < spatial_decay_ && spatial_decay_ <= 1)
      << "Spatial decay must be between 0 and 1, (0, 1].";

  LOG(INFO) << "BTV set with range " << scale_range_
            << " and decay " << spatial_decay_;
}

std::vector<double> BilateralTotalVariationRegularizer::ApplyToImage(
    const double* image_data, const int num_channels) const {

  CHECK_NOTNULL(image_data);

  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> residuals(num_pixels * num_channels);
  for (int channel = 0; channel < num_channels; ++channel) {
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = util::GetPixelIndex(image_size_, channel, row, col);
        residuals[index] = GetBilateralTotalVariation(
            image_data,
            image_size_,
            channel,
            row,
            col,
            scale_range_,
            spatial_decay_);
      }
    }
  }
  return residuals;
}

std::pair<std::vector<double>, std::vector<double>>
BilateralTotalVariationRegularizer::ApplyToImageWithDifferentiation(
    const double* image_data,
    const std::vector<double>& gradient_constants,
    const int num_channels) const {

  CHECK_NOTNULL(image_data);

  const std::vector<double> residuals = ApplyToImage(image_data, num_channels);

  // Compute the gradient.
  // TODO: add some descriptive comments about computing the gradient.
  const int num_pixels = image_size_.width * image_size_.height;
  const int num_parameters = num_pixels * num_channels;
  std::vector<double> gradient(num_parameters);
  for (int channel = 0; channel < num_channels; ++channel) {
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = util::GetPixelIndex(image_size_, channel, row, col);
        // Derivative w.r.t. the pixel itself. Need to consider every pixel in
        // this pixel's range window.
        double didi = 0.0;
        for (int i = 0; i < scale_range_; ++i) {
          for (int j = 0; j < scale_range_; ++j) {
            const int offset_row = row + i;
            const int offset_col = col + j;
            if (offset_row >= image_size_.height ||
                offset_col >= image_size_.width) {
              continue;
            }
            const int offset_index = util::GetPixelIndex(
                image_size_, channel, offset_row, offset_col);
            const double diff = image_data[index] - image_data[offset_index];
            double abs_gradient = 0.0;
            if (diff > 0.0) {
              abs_gradient = 1.0;
            } else if (diff < 0.0) {
              abs_gradient = -1.0;
            }
            const double decay = std::pow(spatial_decay_, i + j);
            didi += decay * abs_gradient;
          }
        }
        gradient[index] +=
            2 * gradient_constants[index] * residuals[index] * didi;
        // Derivative w.r.t. all other pixels where the range window overlaps
        // this pixel.
        for (int i = 0; i < scale_range_; ++i) {
          for (int j = 0; j < scale_range_; ++j) {
            const int offset_row = row - i;
            const int offset_col = col - j;
            if ((offset_row == 0 && offset_col == 0) ||
                 offset_row < 0 || offset_col < 0) {
              continue;
            }
            const int offset_index = util::GetPixelIndex(
                image_size_, channel, offset_row, offset_col);
            const double diff = image_data[offset_index] - image_data[index];
            double didj = 0.0;
            if (diff < 0.0) {
              didj = 1.0;
            } else if (diff > 0.0) {
              didj = -1.0;
            }
            const double decay = std::pow(spatial_decay_, i + j);
            didj *= decay;
            gradient[index] +=
                2 *
                gradient_constants[offset_index] *
                residuals[offset_index] *
                didj;
          }
        }
      }
    }
  }

  return std::make_pair(residuals, gradient);
}

}  // namespace super_resolution
