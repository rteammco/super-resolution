#include <memory>
#include <vector>

#include "image/image_data.h"
#include "solvers/irls_cost_processor.h"
#include "solvers/regularizer.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ContainerEq;
using testing::DoubleEq;
using testing::Each;
using testing::ElementsAre;
using testing::Matcher;
using testing::Return;
using testing::SizeIs;

class MockRegularizer : public super_resolution::Regularizer {
 public:
  // Handle super constructor, since we don't need the image_size_ field.
  MockRegularizer() : super_resolution::Regularizer(cv::Size(0, 0)) {}

  MOCK_CONST_METHOD1(
      ApplyToImage, std::vector<double>(const double* image_data));

  MOCK_CONST_METHOD2(
      GetDerivatives, std::vector<double>(
          const double* image_data, const double* partial_const_terms));
};

// Verifies that the correct data term residuals are returned for an image.
// TODO: fix.
TEST(IrlsCostProcessor, ComputeDataTermResiduals) {
  const cv::Size image_size(3, 3);
  const cv::Mat lr_channel_1 = (cv::Mat_<double>(3, 3)
      << 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5,
         0.5, 0.5, 0.5);
  const cv::Mat lr_channel_2 = (cv::Mat_<double>(3, 3)
      << 1.0,  0.5,  0.0,
         0.25, 0.5,  0.75,
         1.0,  0.0,  1.0);
  const cv::Mat lr_channel_3 = (cv::Mat_<double>(3, 3)
      << 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0,
         1.0, 1.0, 1.0);

  super_resolution::ImageData lr_image_data_1;
  lr_image_data_1.AddChannel(lr_channel_1);
  lr_image_data_1.AddChannel(lr_channel_2);
  super_resolution::ImageData lr_image_data_2(lr_channel_3);
  const std::vector<super_resolution::ImageData> low_res_images = {
    lr_image_data_1,  // 2 channels
    lr_image_data_2   // 1 channel
  };

  super_resolution::ImageModel empty_image_model(2);
  std::unique_ptr<super_resolution::Regularizer> regularizer(
      new MockRegularizer());
  super_resolution::IrlsCostProcessor irls_cost_processor(
      low_res_images,
      empty_image_model,
      image_size,
      std::move(regularizer),
      0.0);  // We're skipping regularization in this test.

  const double hr_pixel_values[9] = {
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5
  };

  // (Image 1, Channel 1) and hr pixels are identical, so expect all zeros.
  std::vector<double> residuals_channel_1 =
      irls_cost_processor.ComputeDataTermResiduals(0, 0, hr_pixel_values);
  EXPECT_THAT(residuals_channel_1, Each(0));

  // (Image 1, Channel 2) residuals should be different at each pixel.
  std::vector<double> residuals_channel_2 =
      irls_cost_processor.ComputeDataTermResiduals(0, 1, hr_pixel_values);
  EXPECT_THAT(residuals_channel_2, ElementsAre(
      -0.5,  0.0,  0.5,
       0.25, 0.0, -0.25,
      -0.5,  0.5, -0.5));

  // (Image 2, channel 1) ("channel_3") should all be -0.5.
  std::vector<double> residuals_channel_3 =
      irls_cost_processor.ComputeDataTermResiduals(1, 0, hr_pixel_values);
  EXPECT_THAT(residuals_channel_3, Each(-0.5));

  // TODO: Mock the ImageModel and make sure the residuals are computed
  // correctly if the HR image is degraded first.
}

// Verifies that the correct regularization residuals are returned for an
// image. This test does not cover regularization operators; instead, it tests
// the IrlsCostProcessor with a mock Regularizer.
TEST(IrlsCostProcessor, ComputeRegularizationResiduals) {
  // Mocked Regularizer.
  std::unique_ptr<MockRegularizer> mock_regularizer(new MockRegularizer());
  const double image_data[5] = {1, 2, 3, 4, 5};
  const std::vector<double> residuals = {1, 2, 3, 4, 5};
  EXPECT_CALL(*mock_regularizer, ApplyToImage(image_data))
      .Times(3)  // 3 calls: compute residuals, update weights, compute again.
      .WillRepeatedly(Return(residuals));

  std::vector<super_resolution::ImageData> empty_image_vector;
  super_resolution::ImageModel empty_image_model(2);

  const double regularization_parameter = 0.5;
  super_resolution::IrlsCostProcessor irls_cost_processor(
      empty_image_vector,
      empty_image_model,
      cv::Size(5, 1),
      std::move(mock_regularizer),
      regularization_parameter);

  // Expected residuals should be the residuals returned by the mocked
  // Regularizer times the regularization parameter and the square root of the
  // respective weights, which are all 1.0 to begin with.
  const Matcher<double> expected_residuals_1[5] = {
    DoubleEq(residuals[0] * regularization_parameter),
    DoubleEq(residuals[1] * regularization_parameter),
    DoubleEq(residuals[2] * regularization_parameter),
    DoubleEq(residuals[3] * regularization_parameter),
    DoubleEq(residuals[4] * regularization_parameter)
  };
  const std::vector<double> returned_residuals_1 =
      irls_cost_processor.ComputeRegularizationResiduals(image_data);
  EXPECT_THAT(returned_residuals_1, ElementsAreArray(expected_residuals_1));

  // Update weights and test again. The weights are expected to be updated as
  // follows:
  //   w = 1.0 / sqrt(residual)
  // so, given residuals [1, 2, 3, 4, 5]:
  //   w0 = 1.0 / 1.0 ~= 1.0
  //   w1 = 1.0 / 2.0 ~= 0.5
  //   w2 = 1.0 / 3.0 ~= 0.333333333
  //   w3 = 1.0 / 4.0 ~= 0.25
  //   w4 = 1.0 / 5.0 ~= 0.2
  //
  // TODO: test with updated weights for a non-L1 norm regularizer.
  irls_cost_processor.UpdateIrlsWeights(image_data);

  // Now expect the residuals to be multiplied by the regularization parameter
  // and the square root of the newly computed weights.
  const Matcher<double> expected_residuals_2[5] = {
    DoubleEq(residuals[0] * regularization_parameter * sqrt(1.0 / 1.0)),
    DoubleEq(residuals[1] * regularization_parameter * sqrt(1.0 / 2.0)),
    DoubleEq(residuals[2] * regularization_parameter * sqrt(1.0 / 3.0)),
    DoubleEq(residuals[3] * regularization_parameter * sqrt(1.0 / 4.0)),
    DoubleEq(residuals[4] * regularization_parameter * sqrt(1.0 / 5.0))
  };
  const std::vector<double> returned_residuals_2 =
      irls_cost_processor.ComputeRegularizationResiduals(image_data);
  EXPECT_THAT(returned_residuals_2, ElementsAreArray(expected_residuals_2));
}
