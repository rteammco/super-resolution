#include "util/test_util.h"

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace super_resolution {
namespace test {

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

  cv::Mat diff;

  // If diff_tolerance is 0 or less, the matrices must match exactly.
  // Otherwise, do a near match with the diff_tolerance value.
  if (diff_tolerance <= 0) {
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
  } else {
    cv::absdiff(mat1, mat2, diff);
    cv::threshold(diff, diff, diff_tolerance, 1, cv::THRESH_BINARY);
  }

  const bool are_equal = (cv::countNonZero(diff) == 0);
  if (!are_equal) {
    std::cout << "Note: matrices are NOT equal:" << std::endl
              << mat1 << std::endl << "--- vs. ---" << std::endl
              << mat2 << std::endl;
    if (diff_tolerance > 0) {
      std::cout << "  >> Diff tolerance of " << diff_tolerance
                << " was exceeded." << std::endl;
    }
  }
  return are_equal;
}

}  // namespace test
}  // namespace super_resolution
