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
    const int border_mode = cv::BORDER_CONSTANT);

// Thresholds a matrix such that any value larger than the max value is reduced
// to the max value and any value smaller than the min value is increased to
// the min value. For example, with min_value = 0.0 and max_value = 1.0, all
// values will be surpressed between 0 and 1. If the image contains multiple
// channels, the threshold will be applied to all channels identically.
void ThresholdImage(
    cv::Mat image, const double min_value, const double max_value);

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_MATRIX_UTIL_H_
