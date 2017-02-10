#include <cmath>
#include <limits>

#include "evaluation/peak_signal_to_noise_ratio.h"
#include "image/image_data.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(Evaluation, PSNR) {
  const cv::Mat ground_truth_matrix = (cv::Mat_<double>(4, 4)
      << 0.0, 0.1, 0.2, 0.3,
         0.7, 0.6, 0.5, 0.4,
         0.8, 0.9, 1.0, 0.5,
         0.4, 0.6, 0.0, 1.0);
  const super_resolution::ImageData ground_truth(ground_truth_matrix);

  const super_resolution::PeakSignalToNoiseRatioEvaluator psnr_evaluator(
      ground_truth);

  /* Verify that when the images are identical, the PSNR is infinity. */

  const super_resolution::ImageData test_image_1 = ground_truth;
  const double psnr_result_1 = psnr_evaluator.Evaluate(test_image_1);
  EXPECT_EQ(psnr_result_1, std::numeric_limits<double>::infinity());

  /* Verify correctness with an explicitly computed small difference. */

  const super_resolution::ImageData test_image_2(ground_truth_matrix);
  // Modify a few of the image pixels:
  double* image_data = test_image_2.GetMutableChannelData(0);
  image_data[6] = 0.25;  // Change from 0.5 to 0.25.
  image_data[15] = 0.5;  // Change from 1.0 to 0.5.
  // Expected MSE is:
  //     (1/16) ((0.5 - 0.25)^2 + (1.0 - 0.5)^2)
  //   = (1/16) (0.25^2 + 0.5^2)
  //   = (1/16) (0.0625 + 0.25)
  //   = (1/16) (0.3125) = 0.0625 * 0.3125
  //   = 0.01953125
  // Expected PSNR is:
  //     10 * log10(1.0^2 / 0.01953125)
  //   = 10 * log10(1.0 / 0.01953125)
  //   = 10 * log10(51.2) = 10 * 1.709269960975831
  //   = 17.09269960975831
  const double expected_psnr_2 = 17.09269960975831;
  const double psnr_result_2 = psnr_evaluator.Evaluate(test_image_2);
  EXPECT_DOUBLE_EQ(psnr_result_2, expected_psnr_2);

  /* Verify correctness on a single channel image with random differences. */

  const cv::Mat test_image_matrix_3 = (cv::Mat_<double>(4, 4)
      << 0.2, 0.9, 1.0, 0.0,
         0.7, 0.0, 0.8, 0.3,
         0.1, 0.0, 0.2, 1.0,
         0.0, 0.5, 0.5, 0.3);
  const super_resolution::ImageData test_image_3(test_image_matrix_3);
  double sum_of_squared_differences = 0.0;
  for (int i = 0; i < 16; ++i) {
    const double diff =
        ground_truth_matrix.at<double>(i) - test_image_matrix_3.at<double>(i);
    sum_of_squared_differences += (diff * diff);
  }
  const double expected_mse_3 = (1.0 / 16.0) * sum_of_squared_differences;
  const double expected_psnr_3 = 10.0 * log10(1.0 / expected_mse_3);
  const double psnr_result_3 = psnr_evaluator.Evaluate(test_image_3);
  EXPECT_DOUBLE_EQ(psnr_result_3, expected_psnr_3);

  /* Verify correctness on a multi-channel image with random differences. */

  // Add 3 channels to the ground truth, all the same for test purposes.
  super_resolution::ImageData ground_truth_multichannel;
  ground_truth_multichannel.AddChannel(ground_truth_matrix);
  ground_truth_multichannel.AddChannel(ground_truth_matrix);
  ground_truth_multichannel.AddChannel(ground_truth_matrix);
  const super_resolution::PeakSignalToNoiseRatioEvaluator psnr_evaluator_2(
      ground_truth_multichannel);

  // For the test image to be evaluated, use the previous three images.
  super_resolution::ImageData test_image_4;
  test_image_4.AddChannel(ground_truth_matrix);
  test_image_4.AddChannel(test_image_2.GetChannelImage(0));
  test_image_4.AddChannel(test_image_matrix_3);

  // Sum of squared differences:
  //   For channel 1 = 0 (the images are identical).
  //   For channel 2 = 0.25^2 + 0.5^2 = 0.3125 (see above computation comments).
  //   For channel 3 = sum_of_squared_differences (computed above).
  // MSE = 1/48 * SSD (since there are 16 * 3 = 48 pixels).
  const double expected_mse_4 =
      (1.0 / 48.0) * (0.0 + 0.3125 + sum_of_squared_differences);
  const double expected_psnr_4 = 10.0 * log10(1.0 / expected_mse_4);
  const double psnr_result_4 = psnr_evaluator_2.Evaluate(test_image_4);
  EXPECT_DOUBLE_EQ(psnr_result_4, expected_psnr_4);
}
