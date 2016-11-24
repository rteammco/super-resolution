#include <memory>
#include <vector>

#include "image_model/additive_noise_module.h"
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
      ApplyToPatch,
      cv::Mat(
          const cv::Mat& patch,
          const int image_index,
          const int channel_index,
          const int pixel_index));

  // We also have to mock this because it's pure virtual.
  MOCK_CONST_METHOD0(GetPixelPatchRadius, int());

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
  // Patch radius for single pixel degradation should be 0 since it only needs
  // the pixel value itself.
  const super_resolution::AdditiveNoiseModule additive_noise_module(5);
  EXPECT_EQ(additive_noise_module.GetPixelPatchRadius(), 0);

  // TODO: implement other tests.
}

TEST(ImageModel, DownsamplingModule) {
  // Patch radius for single pixel degradation should be s/2 where s is the
  // downsampling scale.
  for (int scale = 1; scale <= 5; ++scale) {
    const super_resolution::DownsamplingModule downsampling_module(scale);
    EXPECT_EQ(downsampling_module.GetPixelPatchRadius(), scale / 2);
  }

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

// Tests the implemented functionality of the MotionModule.
// TODO: implement ApplyToImage test.
TEST(ImageModel, MotionModule) {
  /* Verify that the returned patch radius is correct. */

  // Patch radius even for negative values should always be the max abs shift
  // possible. In this case, -2.2 is the largest, so the shift should round up
  // to 3.
  super_resolution::MotionShiftSequence radius_test_motion_shift_sequence({
    super_resolution::MotionShift(1, 1),
    super_resolution::MotionShift(1, -1),
    super_resolution::MotionShift(-2.2, -1)
  });
  const super_resolution::MotionModule radius_test_motion_module(
      radius_test_motion_shift_sequence);
  EXPECT_EQ(radius_test_motion_module.GetPixelPatchRadius(), 3);

  super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(1, 1),
    super_resolution::MotionShift(-1, 0)
  });
  const super_resolution::MotionModule motion_module(motion_shift_sequence);

  // This test motion module should have a patch radius of 1, since 1 is the
  // largest motion shift in either direction.
  EXPECT_EQ(motion_module.GetPixelPatchRadius(), 1);

  /* Verify that ApplyToPatch correctly applies motion as expected. */

  const cv::Mat input_patch_3x3 = (cv::Mat_<double>(3, 3)
      <<  0.4, 0.75, 0.35,
         0.85,  0.9, 0.01,
          0.3, 0.15, 0.55);

  // For motion shift (0, 0), nothing should move. Since radius is 1, expect a
  // single-value matrix containing the middle element, 0.9.
  const cv::Mat expected_patch_1 = (cv::Mat_<double>(1, 1) << 0.9);
  const cv::Mat returned_patch_1 =
      motion_module.ApplyToPatch(input_patch_3x3, 0, 0, 0);
  EXPECT_TRUE(AreMatricesEqual(expected_patch_1, returned_patch_1));

  // For motion shift (1, 1), expect the middle pixel to be shifted down from
  // the top-left corner.
  const cv::Mat expected_patch_2 = (cv::Mat_<double>(1, 1) << 0.4);
  const cv::Mat returned_patch_2 =
      motion_module.ApplyToPatch(input_patch_3x3, 1, 0, 0);
  EXPECT_TRUE(AreMatricesEqual(expected_patch_2, returned_patch_2));

  const cv::Mat input_patch_5x5 = (cv::Mat_<double>(5, 5)
      << 0.33,  0.4, 0.75, 0.35,  0.2,
         0.61, 0.62, 0.63, 0.64, 0.65,
         0.99, 0.85,  0.9, 0.01, 0.78,
          0.1,  0.3, 0.15, 0.55,  0.5,
         0.14, 0.24, 0.34, 0.44, 0.54);

  // Similarly, for (0, 0) expect nothing to move, but this time the returned
  // patch should be 3x3.
  const cv::Mat expected_patch_3 = (cv::Mat_<double>(3, 3)
      << 0.62, 0.63, 0.64,
         0.85,  0.9, 0.01,
          0.3, 0.15, 0.55);
  const cv::Mat returned_patch_3 =
      motion_module.ApplyToPatch(input_patch_5x5, 0, 0, 0);
  EXPECT_TRUE(AreMatricesEqual(expected_patch_3, returned_patch_3));

  // And lastly, for the third shift, verify that we get a shift to the left.
  const cv::Mat expected_patch_4 = (cv::Mat_<double>(3, 3)
      << 0.63, 0.64, 0.65,
          0.9, 0.01, 0.78,
         0.15, 0.55,  0.5);
  const cv::Mat returned_patch_4 =
      motion_module.ApplyToPatch(input_patch_5x5, 2, 0, 0);
  EXPECT_TRUE(AreMatricesEqual(expected_patch_4, returned_patch_4));

  /* Verify that the correct motion operator matrices are returned. */

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
  // Patch radius for single pixel degradation should be the same size as the
  // blur kernel radius. Note that radius must be odd.
  for (int blur_radius = 1; blur_radius <= 9; blur_radius += 2) {
    const double sigma = (blur_radius - 0.5) * 3;  // sigma doesn't matter
    const super_resolution::PsfBlurModule blur_module(blur_radius, sigma);
    EXPECT_EQ(blur_module.GetPixelPatchRadius(), blur_radius);
  }

  // TODO: implement other tests.
}

// Tests that both the ApplyToImage and the ApplyToPixel methods correctly
// return the right values of the degraded image. This does not test the
// method's efficiency, but verifies its correctness and compares the two
// functions to make sure they both return the same values.
TEST(ImageModel, ApplyModel) {
  // TODO: finish implementing this.
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
