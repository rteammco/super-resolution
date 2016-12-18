#include "solvers/tv_regularizer.h"

#include <cmath>
#include <vector>

#include "glog/logging.h"

namespace super_resolution {

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

std::vector<double> TotalVariationRegularizer::ApplyToImage(
    const double* image_data) const {

  CHECK_NOTNULL(image_data);

  std::vector<double> residuals(image_size_.width * image_size_.height);
  for (int row = 0; row < image_size_.height; ++row) {
    for (int col = 0; col < image_size_.width; ++col) {
      const int index = row * image_size_.width + col;
      const int y_variation =
          GetYGradientAtPixel(image_data, image_size_, row, col);
      const int x_variation =
          GetXGradientAtPixel(image_data, image_size_, row, col);
      const double total_variation =
          (y_variation * y_variation) + (x_variation * x_variation);
      residuals[index] = sqrt(total_variation);
    }
  }
  return residuals;
}

std::vector<double> TotalVariationRegularizer::GetDerivatives(
      const double* image_data, const double* partial_const_terms) const {

  CHECK_NOTNULL(image_data);
  CHECK_NOTNULL(partial_const_terms);

  std::vector<double> derivatives(
      image_size_.width * image_size_.height);
  for (int row = 0; row < image_size_.height; ++row) {
    for (int col = 0; col < image_size_.width; ++col) {
      // For pixel at row and col (r, c), the derivative depends on the
      // following pixels:
      //   x_{r,c}    = this pixel itself
      //   x_{r,c-1}  = pixel to the left
      //   x_{r-1,c}  = pixel above

      // Partial w.r.t. x_{r,c} (this pixel) is:
      //   (x_{r,c} - x_{r,c+1}) + (x_{r,c} - x{r+1,c})
      const double this_pixel_partial =
          GetXGradientAtPixel(image_data, image_size_, row, col) -
          GetYGradientAtPixel(image_data, image_size_, row, col);

      // Partial w.r.t. x_{r,c-1} (pixel to the left) is:
      //   (x_{r,c-1} - x{r,c})
      // TODO: / regularizer_values[i]...
      const double left_pixel_partial =
          GetXGradientAtPixel(image_data, image_size_, row, col - 1);

      // Partial w.r.t. x_{r-1,c} (pixel above) is:
      //   (x_{r-1,c} - x_{r,c})
      const double above_pixel_partial =
          GetYGradientAtPixel(image_data, image_size_, row - 1, col);

      // For a reweighted least squares term, the final derivative is
      //   2 * r (
      //     w_{r,c}^2 * this_pixel_partial -
      //     w_{r,c-1}^2 * left_pixel_partial -
      //     w_{r-1,c}^2 * above_pixel_partial)
      // where r is the regularization parameter and w are the weights at each
      // pixel location. 
      // We expect the 2*r*w_{i,j} terms to be passed in as part of the
      // partial_const_terms vector, so we just multiply those into the
      // partials and sum it up.
      const int index = row * image_size_.width + col;
      derivatives[index] = partial_const_terms[index] * this_pixel_partial;
      if (col - 1 >= 0) {  // If left pixel is outside image, the partial is 0.
        const int left_pixel_index = row * image_size_.width + (col - 1);
        derivatives[index] +=
            partial_const_terms[left_pixel_index] * left_pixel_partial;
      }
      if (row - 1 >= 0) {  // If above pixel is outside image, the partial is 0.
        const int above_pixel_index = (row - 1) * image_size_.width + col;
        derivatives[index] +=
            partial_const_terms[above_pixel_index] * above_pixel_partial;
      }
    }
  }
  return derivatives;
}

}  // namespace super_resolution
