#include "util/test_util.h"

#include <iostream>

#include "opencv2/core/core.hpp"

namespace super_resolution {
namespace test {

bool AreMatricesEqual(const cv::Mat& mat1, const cv::Mat& mat2) {
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
  cv::compare(mat1, mat2, diff, cv::CMP_NE);
  const bool are_equal = (cv::countNonZero(diff) == 0);
  if (!are_equal) {
    std::cout << "Note: matrices are NOT equal:" << std::endl
              << mat1 << std::endl << "--- vs. ---" << std::endl
              << mat2 << std::endl;
  }
  return are_equal;
}

}  // namespace test
}  // namespace super_resolution
