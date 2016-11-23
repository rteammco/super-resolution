#include <iostream>
#include <memory>
#include <vector>

#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "image_model/psf_blur_module.h"
#include "motion/motion_shift.h"
#include "util/test_util.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::test::AreMatricesEqual;
using testing::_;
using testing::Return;

const cv::Mat kSmallTestImage = (cv::Mat_<double>(4, 6)
    << 1, 2, 3, 4, 5, 6,
       7, 8, 9, 0, 1, 2,
       9, 7, 5, 4, 2, 1,
       2, 4, 6, 8, 0, 1);
const cv::Size kSmallTestImageSize = cv::Size(6, 4);  // 24 pixels total

// Mock the DegradationOperator
class MockDegradationOperator : public super_resolution::DegradationOperator {
 public:
  // We have to mock this because it's pure virtual.
  MOCK_CONST_METHOD2(
      ApplyToImage,
      void(super_resolution::ImageData* image_data, const int index));

  // We also have to mock this because it's pure virtual.
  MOCK_CONST_METHOD4(
      ApplyToPixel,
      double(
          const super_resolution::ImageData& image_data,
          const int image_index,
          const int channel_index,
          const int pixel_index));

  // Returns a cv::Mat degradation operator in matrix form.
  MOCK_CONST_METHOD2(
      GetOperatorMatrix, cv::Mat(const cv::Size& image_size, const int index));
};

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
  super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(1, 1),
    super_resolution::MotionShift(-1, 0)
  });
  const super_resolution::MotionModule motion_module(motion_shift_sequence);

  // Trivial case: MotionShift(0, 0) should be the identity.
  const cv::Size image_size(3, 3);
  const cv::Mat motion_matrix_1 =
      motion_module.GetOperatorMatrix(image_size, 0);
  const cv::Mat expected_matrix_1 =
      cv::Mat::eye(9, 9, super_resolution::util::kOpenCvMatrixType);
  EXPECT_TRUE(AreMatricesEqual(motion_matrix_1, expected_matrix_1));

  // MotionShift(1, 1) should shift every pixel down and to the right, leaving
  // pixel indices 0, 1, 2, 3, and 6 empty:
  //
  //   | a | b | c |      |   |   |   |
  //   | d | e | f |  =>  |   | a | b |
  //   | g | h | i |      |   | d | e |
  //
  // Hence, given row-first indexing:
  //   'a' moves from index 0 to 4,
  //   'b' moves from index 1 to 5,
  //   'd' moves from index 3 to 7, and
  //   'a' moves from index 4 to 8.
  //
  // The operation matrix represents the pixel value of the output at each
  // pixel index by row; this rows 0, 1, 2, 3, and 6 are all 0 as they map to
  // no pixels in the original image. Row 4 has a 1 in column 0 so it gets
  // the pixel value at index 0 of the original image, and so on.
  const cv::Mat expected_matrix_2 = (cv::Mat_<double>(9, 9)
      << 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0);
  const cv::Mat motion_matrix_2 =
      motion_module.GetOperatorMatrix(image_size, 1);
  EXPECT_TRUE(AreMatricesEqual(motion_matrix_2, expected_matrix_2));

  // MotionShift(-1, 0) shifts the X axis (columns) by -1 as follows:
  //
  //   | a | b | c |      | b | c |   |
  //   | d | e | f |  =>  | e | f |   |
  //   | g | h | i |      | h | i |   |
  //
  // Thus, the expected matrix is as follows:
  const cv::Mat expected_matrix_3 = (cv::Mat_<double>(9, 9)
      << 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0);
  const cv::Mat motion_matrix_3 =
      motion_module.GetOperatorMatrix(image_size, 2);
  EXPECT_TRUE(AreMatricesEqual(motion_matrix_3, expected_matrix_3));
}

TEST(ImageModel, PsfBlurModule) {
  // TODO: implement
}

// Tests that both the ApplyToImage and the ApplyToPixel methods correctly
// return the right values of the degraded image. This does not test the
// method's efficiency, but verifies its correctness and compares the two
// functions to make sure they both return the same values.
TEST(ImageModel, ApplyModel) {
  const super_resolution::ImageData input_image(kSmallTestImage);

  std::unique_ptr<MockDegradationOperator> mock_operator(
      new MockDegradationOperator());
  EXPECT_CALL(*mock_operator, ApplyToImage(_, 0))
      .Times(4);  // TODO: this is the current implementation.

  super_resolution::ImageModel image_model;
  image_model.AddDegradationOperator(std::move(mock_operator));

  const double pixel_0 = image_model.ApplyToPixel(input_image, 0, 0, 0);
  EXPECT_EQ(pixel_0, input_image.GetPixelValue(0, 0));

  const double pixel_1 = image_model.ApplyToPixel(input_image, 0, 0, 1);
  EXPECT_EQ(pixel_1, input_image.GetPixelValue(0, 1));

  const double pixel_4 = image_model.ApplyToPixel(input_image, 0, 0, 4);
  EXPECT_EQ(pixel_4, input_image.GetPixelValue(0, 4));

  const double pixel_9 = image_model.ApplyToPixel(input_image, 0, 0, 9);
  EXPECT_EQ(pixel_9, input_image.GetPixelValue(0, 9));
}

// Tests that the GetModelMatrix method correctly returns the appropriately
// multiplied degradation matrices.
TEST(ImageModel, GetModelMatrix) {
  super_resolution::ImageModel image_model;
  const cv::Size image_size(2, 2);

  const cv::Mat operator_matrix_1 = (cv::Mat_<double>(4, 4)
      << 0, 0, 0, -3,
         4, 3, 2, 1,
         3, 1, 4, 9,
         1, 0, 0, 1);
  std::unique_ptr<MockDegradationOperator> mock_operator_1(
      new MockDegradationOperator());
  EXPECT_CALL(*mock_operator_1, GetOperatorMatrix(image_size, 0))
      .WillOnce(Return(operator_matrix_1));

  const cv::Mat operator_matrix_2 = (cv::Mat_<double>(4, 4)
      << 0, 2, 0, 5,
         1, 1, 1, 1,
         0, 0, 0, 0,
         1, 2, 3, -4);
  std::unique_ptr<MockDegradationOperator> mock_operator_2(
      new MockDegradationOperator());
  EXPECT_CALL(*mock_operator_2, GetOperatorMatrix(image_size, 0))
      .WillOnce(Return(operator_matrix_2));

  const cv::Mat operator_matrix_3 = (cv::Mat_<double>(3, 4)
      << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0);
  std::unique_ptr<MockDegradationOperator> mock_operator_3(
      new MockDegradationOperator());
  EXPECT_CALL(*mock_operator_3, GetOperatorMatrix(image_size, 0))
      .WillOnce(Return(operator_matrix_3));

  image_model.AddDegradationOperator(std::move(mock_operator_1));
  image_model.AddDegradationOperator(std::move(mock_operator_2));
  image_model.AddDegradationOperator(std::move(mock_operator_3));

  // op3 * (op2 * op1)
  const cv::Mat expected_result = (cv::Mat_<double>(3, 4)
      << 13, 6, 4, 7,
          8, 4, 6, 8,
          0, 0, 0, 0);
  cv::Mat returned_operator_matrix = image_model.GetModelMatrix(image_size, 0);
  EXPECT_TRUE(AreMatricesEqual(returned_operator_matrix, expected_result));
}
