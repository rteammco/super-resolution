#include <memory>
#include <vector>

#include "image/image_data.h"
#include "solvers/irls_cost_processor.h"
#include "solvers/regularizer.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ContainerEq;
using testing::Each;
using testing::ElementsAre;
using testing::Return;
using testing::SizeIs;

class MockRegularizer : public super_resolution::Regularizer {
 public:
  // Handle super constructor, since we don't need the image_size_ field.
  MockRegularizer() : super_resolution::Regularizer(cv::Size(0, 0)) {}

  MOCK_CONST_METHOD1(
      ComputeResiduals, std::vector<double>(const double* image_data));
};

// Verifies that the correct data term residuals are returned for an image.
TEST(IrlsCostProcessor, ComputeDataTermResidual) {
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

  super_resolution::ImageModel empty_image_model;
  std::unique_ptr<super_resolution::Regularizer> regularizer(
      new MockRegularizer());
  const std::vector<double> irls_weights(9);  // empty
  super_resolution::IrlsCostProcessor irls_cost_processor(
      low_res_images,
      empty_image_model,
      image_size,
      std::move(regularizer),
      0.0,  // We're skipping regularization in this test.
      &irls_weights);

  const double hr_pixel_values[9] = {
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5
  };

  // (Image 1, Channel 1) and hr pixels are identical, so expect all zeros.
  std::vector<double> residuals_channel_1;
  for (int i = 0; i < 9; ++i) {
    residuals_channel_1.push_back(
        irls_cost_processor.ComputeDataTermResidual(0, 0, i, hr_pixel_values));
  }
  EXPECT_THAT(residuals_channel_1, Each(0));

  // (Image 1, Channel 2) residuals should be different at each pixel.
  std::vector<double> residuals_channel_2;
  for (int i = 0; i < 9; ++i) {
    residuals_channel_2.push_back(
        irls_cost_processor.ComputeDataTermResidual(0, 1, i, hr_pixel_values));
  }
  EXPECT_THAT(residuals_channel_2, ElementsAre(
      -0.5,  0.0,  0.5,
       0.25, 0.0, -0.25,
      -0.5,  0.5, -0.5));

  // (Image 2, channel 1) ("channel_3") should all be -0.5.
  std::vector<double> residuals_channel_3;
  for (int i = 0; i < 9; ++i) {
    residuals_channel_3.push_back(
      irls_cost_processor.ComputeDataTermResidual(1, 0, i, hr_pixel_values));
  }
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
  EXPECT_CALL(*mock_regularizer, ComputeResiduals(image_data))
      .Times(2)  // 2 calls for 2 tests
      .WillRepeatedly(Return(residuals));

  std::vector<super_resolution::ImageData> empty_image_vector;
  super_resolution::ImageModel empty_image_model;

  // The IRLS weights and regularization parameter.
  std::vector<double> irls_weights = {1, 0.5, 0.25, 0.8, 0.0};
  const double regularization_parameter = 0.5;

  super_resolution::IrlsCostProcessor irls_cost_processor(
      empty_image_vector,
      empty_image_model,
      cv::Size(0, 0),
      std::move(mock_regularizer),
      regularization_parameter,
      &irls_weights);

  // Expected residuals should be the residuals returned by the mocked
  // Regularizer times the regularization parameter and the square root of the
  // respective weights.
  const std::vector<double> expected_residuals = {
    residuals[0] * regularization_parameter * sqrt(irls_weights[0]),
    residuals[1] * regularization_parameter * sqrt(irls_weights[1]),
    residuals[2] * regularization_parameter * sqrt(irls_weights[2]),
    residuals[3] * regularization_parameter * sqrt(irls_weights[3]),
    residuals[4] * regularization_parameter * sqrt(irls_weights[4])
  };
  std::vector<double> returned_residuals =
      irls_cost_processor.ComputeRegularizationResiduals(image_data);
  EXPECT_THAT(returned_residuals, ContainerEq(expected_residuals));

  // If we update the weights, we should expect the residuals to be updated.
  // In this case, all residuals should be 0.
  std::fill(irls_weights.begin(), irls_weights.end(), 0);
  returned_residuals =
      irls_cost_processor.ComputeRegularizationResiduals(image_data);
  EXPECT_THAT(returned_residuals, Each(0));
}
