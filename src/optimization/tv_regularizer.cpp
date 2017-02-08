#include "optimization/tv_regularizer.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"

#include "FADBAD++/fadiff.h"

#include "glog/logging.h"

using fadbad::F;  // FADBAD++ forward derivative template.

namespace super_resolution {

// Minimum total variation so we don't divide by zero.
constexpr double kMinTotalVariation = 0.000001;

// For a given image row and col, returns the value of (x_{r,c+1} - x_{r,c}) if
// c+1 is a valid column position, or 0 otherwise. That is, the X-direction
// gradient between the pixel at position index in the data and the pixel
// immediately to its right in the image.
template<typename T>
T GetXGradientAtPixel(
    const T* image_data,
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
template<typename T>
T GetYGradientAtPixel(
    const T* image_data,
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
template<typename T>
T GetTotalVariationAbs(
    const T* image_data,
    const cv::Size& image_size,
    const int row,
    const int col) {

  T y_variation =
      GetYGradientAtPixel<T>(image_data, image_size, row, col);
  if (y_variation < T(0)) {
    y_variation *= T(-1);
  }
  T x_variation =
      GetXGradientAtPixel<T>(image_data, image_size, row, col);
  if (x_variation < T(0)) {
    x_variation *= T(-1);
  }
  return y_variation + x_variation;
}

// Computes the full squared (for a 2-norm) total variation for a pixel at
// (row, col).
template<typename T>
T GetTotalVariationSquared(
    const T* image_data,
    const cv::Size& image_size,
    const int row,
    const int col) {

  const T y_variation =
      GetYGradientAtPixel<T>(image_data, image_size, row, col);
  const T x_variation =
      GetXGradientAtPixel<T>(image_data, image_size, row, col);
  const T total_variation =
      (y_variation * y_variation) + (x_variation * x_variation);
  return total_variation;
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
          residuals[index] = sqrt(GetTotalVariationSquared<double>(
              data_ptr, image_size_, row, col));
        } else {
          residuals[index] = GetTotalVariationAbs<double>(
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
//  std::vector<F<double>> parameters(
//      image_data, image_data + num_parameters);
//  for (int i = 0; i < num_parameters; ++i) {
//    parameters[i].diff(i, num_parameters);
//  }

//  std::vector<double> residual_values(num_parameters);
//  std::vector<F<double>> residuals(num_parameters);
  std::vector<double> residuals(num_parameters);
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
//    const F<double>* data_ptr = parameters.data() + channel_index;
    const double* data_ptr = image_data + channel_index;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = channel_index + (row * image_size_.width + col);
        if (use_two_norm_) {
          // TODO: put back?
//          residuals[index] = fadbad::sqrt(GetTotalVariationSquared<F<double>>(
//              data_ptr, image_size_, row, col));
        } else {
          residuals[index] = GetTotalVariationAbs<double>(
              data_ptr, image_size_, row, col);
//          residuals[index] = GetTotalVariationAbs<F<double>>(
//              data_ptr, image_size_, row, col);
        }
//        residual_values[index] = residuals[index].x();
      }
    }
  }

//  std::vector<double> gradient(num_parameters);  // inits to 0.
//  // For each pixel, its gradient is the sum of partial derivatives at all
//  // other pixels j with respect to pixel i.
//  for (int channel = 0; channel < num_channels_; ++channel) {
//    const int channel_index = channel * num_pixels;
//    for (int row = 0; row < image_size_.height; ++row) {
//      for (int col = 0; col < image_size_.width; ++col) {
//        // To speed things up, we only look at the partial derivatives that
//        // actually get computed (which is, for any pixel, the partial w.r.t.
//        // the pixel to the left and w.r.t. the pixel above).
//        const int index = channel_index + (row * image_size_.width + col);
//
//        // Add partial w.r.t. x_{r,c} (this pixel):
//        const double didi = residuals[index].d(index);
//        if (!isnan(didi)) {
//          gradient[index] +=
//              2 * gradient_constants[index] * residuals[index].x() * didi;
//        }
//
//        // Partial w.r.t. x_{r,c-1} (pixel to the left):
//        if (col - 1 >= 0) {
//          const int left_index =
//              channel_index + (row * image_size_.width + (col - 1));
//          const double dldi = residuals[left_index].d(index);
//          if (!isnan(dldi)) {
//            gradient[index] +=
//                2 *
//                gradient_constants[left_index] *
//                residuals[left_index].x() *
//                dldi;
//          }
//        }
//
//        // Partial w.r.t. x_{r-1,c} (pixel above):
//        if (row - 1 >= 0) {
//          const int above_index =
//              channel_index + ((row - 1) * image_size_.width + col);
//          const double dadi = residuals[above_index].d(index);
//          if (!isnan(dadi)) {
//            gradient[index] +=
//                2 *
//                gradient_constants[above_index] *
//                residuals[above_index].x() *
//                dadi;
//          }
//        }
//      }
//    }
//  }

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
            GetXGradientAtPixel<double>(data_ptr, image_size_, row, col);
        if (x_gradient < 0) {
          didi += 1.0;
        } else if (x_gradient > 0) {
          didi -= 1.0;
        }
        const double y_gradient =
            GetYGradientAtPixel<double>(data_ptr, image_size_, row, col);
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
              GetXGradientAtPixel<double>(data_ptr, image_size_, row, col - 1);
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
              GetYGradientAtPixel<double>(data_ptr, image_size_, row - 1, col);
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
