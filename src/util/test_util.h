// Contains functions useful for unit testing.

#ifndef SRC_UTIL_TEST_UTIL_H_
#define SRC_UTIL_TEST_UTIL_H_

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {
namespace test {

// Returns true if the two given matrices contain identical values. If
// diff_tolerance is greater than 0, a near-equality will be computed given the
// tolerance value.
// Source:
//   http://stackoverflow.com/questions/9905093/how-to-check-whether-two-matrixes-are-identical-in-opencv  NOLINT
bool AreMatricesEqual(
    const cv::Mat& mat1,
    const cv::Mat& mat2,
    const double diff_tolerance = 0.0);

// Same as AreMatricesEqual, but crops the matrices to exclude a border of the
// given size. This allows for checking images for equal pixel values excluding
// certain border case issues.
bool AreMatricesEqualCroppedBorder(
    const cv::Mat& mat1,
    const cv::Mat& mat2,
    const int crop_border_size,
    const double diff_tolerance = 0);

// Applies the AreMatricesEqual function to each channel of the given images.
bool AreImagesEqual(
    const ImageData& image1,
    const ImageData& image2,
    const double diff_tolerance = 0.0);

}  // namespace test
}  // namespace super_resolution

#endif  // SRC_UTIL_TEST_UTIL_H_
