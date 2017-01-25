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

// Computes the full total variation for a pixel at (row, col) of the given
// image_data. Returned value will be 0 for invalid row and col values.
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
        residuals[index] = sqrt(GetTotalVariationSquared<double>(
            data_ptr, image_size_, row, col));
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
  std::vector<F<double>> parameters(
      image_data, image_data + num_parameters);
  for (int i = 0; i < num_parameters; ++i) {
    parameters[i].diff(i, num_parameters);
  }

  std::vector<double> residual_values(num_parameters);
  std::vector<F<double>> residuals(num_parameters);
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
    const F<double>* data_ptr = parameters.data() + channel_index;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int index = channel_index + (row * image_size_.width + col);
        residuals[index] = fadbad::sqrt(GetTotalVariationSquared<F<double>>(
            data_ptr, image_size_, row, col));
        residual_values[index] = residuals[index].x();
      }
    }
  }

  std::vector<double> gradient(num_parameters);  // inits to 0.
  // For each pixel, its gradient is the sum of partial derivatives at all
  // other pixels j with respect to pixel i.
  for (int channel = 0; channel < num_channels_; ++channel) {
    const int channel_index = channel * num_pixels;
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        // To speed things up, we only look at the partial derivatives that
        // actually get computed (which is, for any pixel, the partial w.r.t.
        // the pixel to the left and w.r.t. the pixel above).
        const int index = channel_index + (row * image_size_.width + col);

        // Add partial w.r.t. x_{r,c} (this pixel):
        const double didi = residuals[index].d(index);
        if (!isnan(didi)) {
          gradient[index] +=
              2 * gradient_constants[index] * residuals[index].x() * didi;
        }

        // Partial w.r.t. x_{r,c-1} (pixel to the left):
        if (col - 1 >= 0) {
          const int left_index =
              channel_index + (row * image_size_.width + (col - 1));
          const double dldi = residuals[left_index].d(index);
          if (!isnan(dldi)) {
            gradient[index] +=
                2 *
                gradient_constants[left_index] *
                residuals[left_index].x() *
                dldi;
          }
        }

        // Partial w.r.t. x_{r-1,c} (pixel above):
        if (row - 1 >= 0) {
          const int above_index =
              channel_index + ((row - 1) * image_size_.width + col);
          const double dadi = residuals[above_index].d(index);
          if (!isnan(dadi)) {
            gradient[index] +=
                2 *
                gradient_constants[above_index] *
                residuals[above_index].x() *
                dadi;
          }
        }
      }
    }
  }

//  // TODO: remove this sanity check.
//  std::vector<double> gradient_2(num_parameters);
//  for (int i = 0; i < num_parameters; ++i) {
//    for (int j = 0; j < num_parameters; ++j) {
//      const double djdi = residuals[j].d(i);
//      if (!isnan(djdi)) {  // If this partial exists...
//        const double gradient_ij =
//            2 * gradient_constants[j] * residuals[j].x() * djdi;
//        gradient_2[i] += gradient_ij;
//      }
//    }
//  }
//  for (int i = 0; i < num_parameters; ++i) {
//    const double absdiff = std::abs(gradient[i] - gradient_2[i]);
//    if (absdiff > 0.00001) {
//      LOG(INFO) << "Mismatch at " << i << ": "
//                << gradient[i] << " vs. " << gradient_2[i];
//    }
//  }

  return std::make_pair(residual_values, gradient);
}

std::vector<double> TotalVariationRegularizer::GetGradient(
    const double* image_data,
    const std::vector<double>& gradient_constants) const {

  CHECK_NOTNULL(image_data);

  const int num_pixels = image_size_.width * image_size_.height;
  CHECK_EQ(gradient_constants.size(), num_pixels)
    << "There must be exactly one const term per pixel in the image. "
    << "Use 1 for identity or 0 to ignore the derivative.";

  const std::vector<double> total_variation = ApplyToImage(image_data);
  // Convert all values to non-zero to avoid division by zero.
  std::vector<double> total_variation_nz;
  for (const double tv_value : total_variation) {
    total_variation_nz.push_back(std::max(tv_value, kMinTotalVariation));
  }

  std::vector<double> derivatives(num_pixels);
  for (int row = 0; row < image_size_.height; ++row) {
    for (int col = 0; col < image_size_.width; ++col) {
      // For pixel at row and col (r, c), the derivative depends on the
      // following pixels:
      //   x_{r,c}    = this pixel itself
      //   x_{r,c-1}  = pixel to the left
      //   x_{r-1,c}  = pixel above
      // All the partials are also later divided by the total_variation value
      // at their respective pixel locations.

      // Partial w.r.t. x_{r,c} (this pixel) is:
      //   ((x_{r,c+1} - x_{r,c}) + (x_{r+1,c} - x{r,c})) / tv_{r,c}
      // We do the tv_{r,c} division later.
      const double this_pixel_numerator =
          GetXGradientAtPixel<double>(image_data, image_size_, row, col) +
          GetYGradientAtPixel<double>(image_data, image_size_, row, col);

      // Partial w.r.t. x_{r,c-1} (pixel to the left) is:
      //   -(x_{r,c} - x{r,c-1}) / tv_{r,c-1}
      const double left_pixel_numerator =
          -GetXGradientAtPixel<double>(image_data, image_size_, row, col - 1);

      // Partial w.r.t. x_{r-1,c} (pixel above) is:
      //   -(x_{r,c} - x_{r-1,c}) / tv_{r-1,c}
      const double above_pixel_numerator =
          -GetYGradientAtPixel<double>(image_data, image_size_, row - 1, col);

      // Divide the tv values and multiply by the given partial constants, and
      // sum it all up to get the final derivative for the pixel at (row, col).

      // Add partial w.r.t. x_{r,c} (this pixel):
      const int this_pixel_index = row * image_size_.width + col;
      const double this_pixel_partial =
          this_pixel_numerator / total_variation_nz[this_pixel_index];
      derivatives[this_pixel_index] =
          gradient_constants[this_pixel_index] * this_pixel_partial;

      // Partial w.r.t. x_{r,c-1} (pixel to the left):
      if (col - 1 >= 0) {  // If left pixel is outside image, the partial is 0.
        const int left_pixel_index = row * image_size_.width + (col - 1);
        const double left_pixel_partial =
            left_pixel_numerator / total_variation_nz[left_pixel_index];
        derivatives[this_pixel_index] +=
            gradient_constants[left_pixel_index] * left_pixel_partial;
      }

      // Partial w.r.t. x_{r-1,c} (pixel above):
      if (row - 1 >= 0) {  // If above pixel is outside image, the partial is 0.
        const int above_pixel_index = (row - 1) * image_size_.width + col;
        const double above_pixel_partial =
            above_pixel_numerator / total_variation_nz[above_pixel_index];
        derivatives[this_pixel_index] +=
            gradient_constants[above_pixel_index] * above_pixel_partial;
      }
    }
  }
  return derivatives;
}

}  // namespace super_resolution
