#include "optimization/tv_regularizer.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "util/util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// For a given image row and col, returns the value of (x_{r,c+1} - x_{r,c}) if
// c+1 is a valid column position, or 0 otherwise. That is, the X-direction
// gradient between the pixel at position index in the data and the pixel
// immediately to its right in the image.
double GetXGradientAtPixel(
    const double* image_data,
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col) {

  if (col >= 0 && col + 1 < image_size.width) {
    const int pixel_index = util::GetPixelIndex(image_size, channel, row, col);
    const int x_neighbor_index =
        util::GetPixelIndex(image_size, channel, row, col + 1);
    return image_data[x_neighbor_index] - image_data[pixel_index];
  }
  return 0;
}

// Same as GetXGradientAtPixel, but for the value below the given pixel
// rather than to the right. That is, the Y-direction gradient at that pixel.
double GetYGradientAtPixel(
    const double* image_data,
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col) {

  if (row >= 0 && row + 1 < image_size.height) {
    const int pixel_index = util::GetPixelIndex(image_size, channel, row, col);
    const int y_neighbor_index =
        util::GetPixelIndex(image_size, channel, row + 1, col);
    return image_data[y_neighbor_index] - image_data[pixel_index];
  }
  return 0;
}

// Same as GetXGradientAtPixel, but uses the Z-direction for multispectral
// images. This does not check for the validity of the index at the next image
// channel, so only call if there is definately a channel + 1.
double GetZGradientAtPixel(
    const double* image_data,
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col) {

  const int pixel_index = util::GetPixelIndex(image_size, channel, row, col);
  const int z_neighbor_index =
      util::GetPixelIndex(image_size, channel + 1, row, col);
  return image_data[z_neighbor_index] - image_data[pixel_index];
}

// Computes the full total variation 1-norm for a pixel at (row, col) of the
// given image_data. Returned value will be 0 for invalid row and col values.
double GetTotalVariationAbs(
    const double* image_data,
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col) {

  const double y_variation = std::abs(
      GetYGradientAtPixel(image_data, image_size, channel, row, col));
  const double x_variation = std::abs(
      GetXGradientAtPixel(image_data, image_size, channel, row, col));
  return y_variation + x_variation;
}

// Computes the 3D total variation, which is defined just as the 2D total
// variation plus the Z-direction (spectral) variation. Same as
// GetTotalVariationAbs if (channel + 1) >= num_channels.
double GetTotalVariation3d(
    const double* image_data,
    const cv::Size& image_size,
    const int num_channels,
    const int channel,
    const int row,
    const int col) {

  double total_variation =
      GetTotalVariationAbs(image_data, image_size, channel, row, col);
  if ((channel + 1) < num_channels) {
    const double z_variation = std::abs(
        GetZGradientAtPixel(image_data, image_size, channel, row, col));
    total_variation += z_variation;
  }
  return total_variation;
}

std::vector<double> TotalVariationRegularizer::ApplyToImage(
    const double* image_data) const {

  CHECK_NOTNULL(image_data);

  const int num_pixels = image_size_.area();
  std::vector<double> residuals(num_pixels * num_channels_);
  for (int channel = 0; channel < num_channels_; ++channel) {
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = util::GetPixelIndex(image_size_, channel, row, col);
        if (use_3d_total_variation_) {
          residuals[index] = GetTotalVariation3d(
              image_data, image_size_, num_channels_, channel, row, col);
        } else {
          residuals[index] = GetTotalVariationAbs(
              image_data, image_size_, channel, row, col);
        }
      }
    }
  }
  return residuals;
}

std::pair<std::vector<double>, std::vector<double>>
TotalVariationRegularizer::ApplyToImageWithDifferentiation(
    const double* image_data,
    const std::vector<double>& gradient_constants) const {

  CHECK_NOTNULL(image_data);

  const std::vector<double> residuals = ApplyToImage(image_data);

  // Compute the gradient.
  // TODO: add some descriptive comments about computing the gradient.
  const int num_pixels = image_size_.area();
  const int num_parameters = num_pixels * num_channels_;
  std::vector<double> gradient(num_parameters);
  for (int channel = 0; channel < num_channels_; ++channel) {
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = util::GetPixelIndex(image_size_, channel, row, col);
        // Derivative w.r.t. the pixel itself.
        double didi = 0.0;
        const double x_gradient =
            GetXGradientAtPixel(image_data, image_size_, channel, row, col);
        if (x_gradient < 0.0) {
          didi += 1.0;
        } else if (x_gradient > 0.0) {
          didi -= 1.0;
        }
        const double y_gradient =
            GetYGradientAtPixel(image_data, image_size_, channel, row, col);
        if (y_gradient < 0.0) {
          didi += 1.0;
        } else if (y_gradient > 0.0) {
          didi -= 1.0;
        }
        gradient[index] +=
            2 * gradient_constants[index] * residuals[index] * didi;
        // Derivative w.r.t. the pixel to the left.
        if (col - 1 >= 0) {
          const int left_index =
              util::GetPixelIndex(image_size_, channel, row, col - 1);
          const double left_gradient = GetXGradientAtPixel(
              image_data, image_size_, channel, row, col - 1);
          double dldi = 0.0;
          if (left_gradient > 0.0) {
            dldi = 1.0;
          } else if (left_gradient < 0.0) {
            dldi = -1.0;
          }
          gradient[index] +=
              2 * gradient_constants[left_index] * residuals[left_index] * dldi;
        }
        // Derivative w.r.t. the pixel above.
        if (row - 1 >= 0) {
          const int above_index =
              util::GetPixelIndex(image_size_, channel, row - 1, col);
          const double above_gradient = GetYGradientAtPixel(
              image_data, image_size_, channel, row - 1, col);
          double dadi = 0.0;
          if (above_gradient > 0.0) {
            dadi = 1.0;
          } else if (above_gradient < 0.0) {
            dadi = -1.0;
          }
          gradient[index] +=
              2 *
              gradient_constants[above_index] *
              residuals[above_index] *
              dadi;
        }
        // Derivative w.r.t. the pixel in the channel before (if 3D TV).
        if (use_3d_total_variation_ && channel > 0) {
          const int before_index =
              util::GetPixelIndex(image_size_, channel - 1, row, col);
          const double before_gradient = GetZGradientAtPixel(
              image_data, image_size_, channel - 1, row, col);
          double dbdi = 0.0;
          if (before_gradient > 0.0) {
            dbdi = 1.0;
          } else if (before_gradient < 0.0) {
            dbdi = -1.0;
          }
          gradient[index] +=
              2 *
              gradient_constants[before_index] *
              residuals[before_index] *
              dbdi;
        }
      }
    }
  }

  return std::make_pair(residuals, gradient);
}

}  // namespace super_resolution
