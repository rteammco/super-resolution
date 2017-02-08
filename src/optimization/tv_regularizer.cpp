#include "optimization/tv_regularizer.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// Minimum total variation so we don't divide by zero.
constexpr double kMinTotalVariation = 0.000001;

// For a given image row and col, returns the value of (x_{r,c+1} - x_{r,c}) if
// c+1 is a valid column position, or 0 otherwise. That is, the X-direction
// gradient between the pixel at position index in the data and the pixel
// immediately to its right in the image.
double GetXGradientAtPixel(
    const double* image_data,
    const cv::Size& image_size,
    const int row,
    const int col) {

  if (col >= 0 && col + 1 < image_size.width) {
    const int pixel_index = row * image_size.width + col;
    const int x_neighbor_index = row * image_size.width + (col + 1);
    return image_data[x_neighbor_index] - image_data[pixel_index];
  }
  return 0;
}

// Same as GetRightPixelDifference, but for the value below the given pixel
// rather than to the right. That is, the Y-direction gradient at that pixel.
double GetYGradientAtPixel(
    const double* image_data,
    const cv::Size& image_size,
    const int row,
    const int col) {

  if (row >= 0 && row + 1 < image_size.height) {
    const int pixel_index = row * image_size.width + col;
    const int y_neighbor_index = (row + 1) * image_size.width + col;
    return image_data[y_neighbor_index] - image_data[pixel_index];
  }
  return 0;
}

// Computes the full total variation 1-norm for a pixel at (row, col) of the
// given image_data. Returned value will be 0 for invalid row and col values.
double GetTotalVariationAbs(
    const double* image_data,
    const cv::Size& image_size,
    const int row,
    const int col) {

  const double y_variation =
      std::abs(GetYGradientAtPixel(image_data, image_size, row, col));
  const double x_variation =
      std::abs(GetXGradientAtPixel(image_data, image_size, row, col));
  return y_variation + x_variation;
}

// Computes the full squared (for a 2-norm) total variation for a pixel at
// (row, col).
double GetTotalVariationSquared(
    const double* image_data,
    const cv::Size& image_size,
    const int row,
    const int col) {

  const double y_variation =
      GetYGradientAtPixel(image_data, image_size, row, col);
  const double x_variation =
      GetXGradientAtPixel(image_data, image_size, row, col);
  const double total_variation_squared =
      (y_variation * y_variation) + (x_variation * x_variation);
  return total_variation_squared;
}

std::vector<double> TotalVariationRegularizer::ApplyToImage(
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
        if (use_two_norm_) {
          residuals[index] = sqrt(GetTotalVariationSquared(
              data_ptr, image_size_, row, col));
        } else {
          residuals[index] = GetTotalVariationAbs(
              data_ptr, image_size_, row, col);
        }
      }
    }
  }
  return residuals;
}

std::pair<std::vector<double>, std::vector<double>>
TotalVariationRegularizer::ApplyToImageWithDifferentiation(
    const double* image_data,
    const std::vector<double>& gradient_constants,
    const GradientComputationMethod& differentiation_method) const {

  CHECK_NOTNULL(image_data);

  // Initialize the derivatives of each parameter with respect to itself.
  const int num_pixels = image_size_.width * image_size_.height;
  const int num_parameters = num_pixels * num_channels_;
  std::vector<double> residuals(num_parameters);
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
    const double* data_ptr = image_data + channel_index;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = channel_index + (row * image_size_.width + col);
        if (use_two_norm_) {
          // TODO: put back?
//          residuals[index] = sqrt(GetTotalVariationSquared(
//              data_ptr, image_size_, row, col));
        } else {
          residuals[index] = GetTotalVariationAbs(
              data_ptr, image_size_, row, col);
        }
      }
    }
  }

  // Compute the gradient.
  // TODO: add some descriptive comments about computing the gradient.
  // TODO: if we're going to keep the 2-norm gradient version, those need a
  //       different gradient computation implementation.
  std::vector<double> gradient(num_parameters);
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
    const double* data_ptr = image_data + channel_index;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = channel_index + (row * image_size_.width + col);
        // Derivative w.r.t. the pixel itself.
        double didi = 0.0;
        const double x_gradient =
            GetXGradientAtPixel(data_ptr, image_size_, row, col);
        if (x_gradient < 0) {
          didi += 1.0;
        } else if (x_gradient > 0) {
          didi -= 1.0;
        }
        const double y_gradient =
            GetYGradientAtPixel(data_ptr, image_size_, row, col);
        if (y_gradient < 0) {
          didi += 1.0;
        } else if (y_gradient > 0) {
          didi -= 1.0;
        }
        gradient[index] +=
            2 * gradient_constants[index] * residuals[index] * didi;
        // Derivative w.r.t. the pixel to the left.
        if (col - 1 >= 0) {
          const int left_index =
              channel_index + (row * image_size_.width + (col - 1));
          const double left_gradient =
              GetXGradientAtPixel(data_ptr, image_size_, row, col - 1);
          double didl = 1.0;
          if (left_gradient < 0) {
            didl = -1.0;
          }
          gradient[index] +=
              2 * gradient_constants[left_index] * residuals[left_index] * didl;
        }
        // Derivative w.r.t. the pixel above.
        if (row - 1 >= 0) {
          const int above_index =
              channel_index + ((row - 1) * image_size_.width + col);
          const double above_gradient =
              GetYGradientAtPixel(data_ptr, image_size_, row - 1, col);
          double dida = 1.0;
          if (above_gradient < 0) {
            dida = -1.0;
          }
          gradient[index] +=
              2 *
              gradient_constants[above_index] *
              residuals[above_index] *
              dida;
        }
      }
    }
  }

  return std::make_pair(residuals, gradient);
}

}  // namespace super_resolution
