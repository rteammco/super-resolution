// Utilities for common matrix operations.

#ifndef SRC_UTIL_MATRIX_UTIL_H_
#define SRC_UTIL_MATRIX_UTIL_H_

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {
namespace util {

// This is the OpenCV matrix format that every matrix should use.
constexpr int kOpenCvMatrixType = CV_64FC1;

// Applies a 2D convolution to the given ImageData. The convolution is applied
// independently to all channels of the image. Specify border mode as needed.
void ApplyConvolutionToImage(
    ImageData* image_data,
    const cv::Mat& kernel,
    const int border_mode = cv::BORDER_REFLECT_101);

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_MATRIX_UTIL_H_
