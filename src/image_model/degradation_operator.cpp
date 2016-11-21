#include "image_model/degradation_operator.h"

#include <utility>
#include <vector>

#include "util/util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// Size limitations for the ConvertKernelToOperator method.
constexpr int kMaxConvolutionImageSize = 30;
constexpr int kMaxConvolutionKernelSize = 10;

cv::Mat DegradationOperator::ConvertKernelToOperatorMatrix(
    const cv::Mat& kernel, const cv::Size& image_size) {

  const cv::Size kernel_size = kernel.size();
  CHECK_LE(kernel_size.width, kMaxConvolutionKernelSize)
      << "Kernel is too big to convert to matrix form.";
  CHECK_LE(kernel_size.height, kMaxConvolutionKernelSize)
      << "Kernel is too big to convert to matrix form.";
  CHECK_LE(image_size.width, kMaxConvolutionImageSize)
      << "Image is too big to compute a kernel matrix.";
  CHECK_LE(image_size.height, kMaxConvolutionImageSize)
      << "Image is too big to compute a kernel matrix.";

  // Initialize a zero matrix for the operator.
  const int num_pixels = image_size.width * image_size.height;
  cv::Mat operator_matrix =
      cv::Mat::zeros(num_pixels, num_pixels, util::kOpenCvMatrixType);

  // Compute kernel offsets. These are all the relative indices where the
  // convolution kernel intersects the 2D image at every pixel.
  const int kernel_mid_row = kernel_size.height / 2;
  const int kernel_mid_col = kernel_size.width / 2;
  std::vector<std::pair<int, int>> kernel_offsets;
  for (int i = 0; i < kernel_size.height; ++i) {
    for (int j = 0; j < kernel_size.width; ++j) {
      kernel_offsets.push_back(
          std::make_pair(i - kernel_mid_row, j - kernel_mid_col));
    }
  }

  // Finally, compute the matrix by computing the convolution intersection
  // indices for every pixel in the image.
  int next_row = 0;  // Next row to set in the resulting matrix.
  for (int row = 0; row < image_size.height; ++row) {
    for (int col = 0; col < image_size.width; ++col) {
      for (const std::pair<int, int>& offset : kernel_offsets) {
        // Find image row and col coordinates where this offset (value) in the
        // kernel intersects with the image.
        const int image_row = row + offset.first;
        const int image_col = col + offset.second;
        if ((image_row >= 0) && (image_row < image_size.height) &&
            (image_col >= 0) && (image_col < image_size.width)) {
          const int kernel_row = offset.first + kernel_mid_row;
          const int kernel_col = offset.second + kernel_mid_col;
          const int image_index = image_row * image_size.width + image_col;
          operator_matrix.at<double>(next_row, image_index) =
              kernel.at<double>(kernel_row, kernel_col);
        }
      }
      next_row++;
    }
  }
  return operator_matrix;
}

cv::Mat DegradationOperator::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_pixels = image_size.width * image_size.height;
  return cv::Mat::eye(num_pixels, num_pixels, util::kOpenCvMatrixType);
}

}  // namespace super_resolution
