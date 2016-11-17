#include <iostream>

#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/psf_blur_module.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

const cv::Mat kSmallTestImage = (cv::Mat_<double>(4, 6)
    << 1, 2, 3, 4, 5, 6,
       7, 8, 9, 0, 1, 2,
       9, 7, 5, 4, 2, 1,
       2, 4, 6, 8, 0, 1);
const cv::Size kSmallTestImageSize = cv::Size(6, 4);  // 24 pixels total

// Returns true if the two given matrices contain identical values.
// Source:
//   http://stackoverflow.com/questions/9905093/how-to-check-whether-two-matrixes-are-identical-in-opencv  NOLINT
// TODO: this may need to do float comparisons.
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

// Tests the static function(s) in DegradationOperator.
TEST(ImageModel, DegradationOperator) {
  const cv::Mat kernel = (cv::Mat_<double>(3, 3)
      << -1, 0, 1,
         -2, 0, 2,
         -1, 0, 1);
  const cv::Mat test_image = (cv::Mat_<double>(2, 3)
      << 1, 3, 5,
         9, 5, 2);
  const cv::Mat operator_matrix =
      super_resolution::DegradationOperator::ConvertKernelToOperatorMatrix(
          kernel, test_image.size());

  // Make sure we get the correct kernel.
  const cv::Mat expected_matrix = (cv::Mat_<double>(6, 6)
      << 0,  2,  0,  0,  1,  0,
         -2, 0,  2,  -1, 0,  1,
         0,  -2, 0,  0,  -1, 0,
         0,  1,  0,  0,  2,  0,
         -1, 0,  1,  -2, 0,  2,
         0,  -1, 0,  0,  -2, 0);
  EXPECT_TRUE(AreMatricesEqual(operator_matrix, expected_matrix));

  // Now make sure that we get the correct image after multiplication.
  const cv::Mat test_image_vector = test_image.reshape(1, 6);
  const cv::Mat expected_result = (cv::Mat_<double>(6, 1)
    << 11, 1,   -11,
       13, -10, -13);
  EXPECT_TRUE(AreMatricesEqual(
      operator_matrix * test_image_vector, expected_result));
}

TEST(ImageModel, AdditiveNoiseModule) {
  // TODO: implement
}

TEST(ImageModel, DownsamplingModule) {
  super_resolution::DownsamplingModule downsampling_module(2);
  const cv::Mat downsampling_matrix =
      downsampling_module.GetOperatorMatrix(kSmallTestImageSize, 0);

  // 24 pixels in high-res input, 6 (= 24 / 2*2) pixels in downsampled output.
  const cv::Mat expected_matrix = (cv::Mat_<double>(6, 24) <<
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);

  EXPECT_TRUE(AreMatricesEqual(downsampling_matrix, expected_matrix));

  // Vectorize the test image and compare to the expected outcome.
  const cv::Mat test_image_vector = kSmallTestImage.reshape(1, 24);
  const cv::Mat expected_downsampled_vector = (cv::Mat_<double>(6, 1)
      << 1, 3, 5,
         9, 5, 2);
  EXPECT_TRUE(AreMatricesEqual(
      downsampling_matrix * test_image_vector, expected_downsampled_vector));
}

TEST(ImageModel, MotionModule) {
  // TODO: implement
}

TEST(ImageModel, PsfBlurModule) {
  // TODO: implement
}

TEST(ImageModel, GetModelMatrix) {
  // TODO: implement, check that the whole ImageModel can return the correct
  // degradation operator matrix.
}
