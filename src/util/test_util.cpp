#include "util/test_util.h"

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace test {

// Maximum matrix size (either width or height) to print. Otherwise, it will be
// considered too big to display.
constexpr int kMaxMatrixSizeToPrint = 15;

bool AreMatricesEqual(
    const cv::Mat& mat1, const cv::Mat& mat2, const double diff_tolerance) {

  if (mat1.empty() && mat2.empty()) {
    return true;
  }
  if (mat1.cols != mat2.cols ||
      mat1.rows != mat2.rows ||
      mat1.dims != mat2.dims) {
    std::cout << "Matrices have different dimensions: "
              << mat1.size() << " vs. " << mat2.size() << std::endl;
    return false;
  }

  // If diff_tolerance is 0 or less, the matrices must match exactly.
  // Otherwise, do a near match with the diff_tolerance value.
  cv::Mat diff;
  if (diff_tolerance <= 0) {
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    diff = cv::abs(diff);
  } else {
    cv::absdiff(mat1, mat2, diff);
    cv::threshold(diff, diff, diff_tolerance, 1, cv::THRESH_BINARY);
  }

  const int non_equal_count = cv::countNonZero(diff);
  const bool are_equal = (non_equal_count == 0);
  if (!are_equal) {
    const cv::Size matrix_size = mat1.size();
    std::cout << "Note: matrices are NOT equal:" << std::endl;
    // Show the matrices if they're small enough to be printed.
    if (matrix_size.width <= kMaxMatrixSizeToPrint &&
        matrix_size.height <= kMaxMatrixSizeToPrint) {
      std::cout << mat1 << std::endl << "--- vs. ---" << std::endl
                << mat2 << std::endl;
    } else {
      std::cout << "  >> Matrices are too large to be displayed." << std::endl;
    }
    if (diff_tolerance > 0) {
      std::cout << "  >> Diff tolerance of " << diff_tolerance
                << " was exceeded." << std::endl;
    }
    const int num_values = mat1.size().width * mat1.size().height;
    std::cout << "  >> Error in " << non_equal_count
              << " values out of " << num_values << "." << std::endl;
    cv::Point ignored_loc, max_loc;
    double ignored_diff, max_diff;
    cv::minMaxLoc(diff, &ignored_diff, &max_diff, &ignored_loc, &max_loc);
    std::cout << "  >> The largest difference was " << max_diff
              << " at position " << max_loc << "." << std::endl;
  }
  return are_equal;
}

bool AreMatricesEqualCroppedBorder(
    const cv::Mat& mat1,
    const cv::Mat& mat2,
    const int crop_border_size,
    const double diff_tolerance) {

  CHECK_GE(crop_border_size, 0);

  const cv::Size size = mat1.size();
  const cv::Rect region_of_interest(
      crop_border_size,                     // Left index of crop.
      crop_border_size,                     // Top index of crop.
      size.width - crop_border_size * 2,    // Width of crop.
      size.height - crop_border_size * 2);  // Height of crop.

  const cv::Mat cropped_mat1 = mat1(region_of_interest);
  const cv::Mat cropped_mat2 = mat2(region_of_interest);

  return AreMatricesEqual(cropped_mat1, cropped_mat2, diff_tolerance);
}

}  // namespace test
}  // namespace super_resolution
