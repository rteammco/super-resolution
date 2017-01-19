#include <limits>

#include "evaluation/peak_signal_to_noise_ratio.h"
#include "image/image_data.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(Evaluation, PSNR) {
  const cv::Mat ground_truth_matrix = (cv::Mat_<double>(4, 4)
      << 0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0);
  const super_resolution::ImageData ground_truth(ground_truth_matrix);

  const super_resolution::PeakSignalToNoiseRatioEvaluator psnr_evaluator(
      ground_truth);

  // If the image is identical to the ground truth, the PSNR should be infinity.
  const super_resolution::ImageData test_image_1 = ground_truth;
  const double psnr_result_1 = psnr_evaluator.Evaluate(test_image_1);
  EXPECT_EQ(psnr_result_1, std::numeric_limits<double>::infinity());

  // Check a single channel. TODO: implement.
  const cv::Mat test_image_matrix_2 = (cv::Mat_<double>(4, 4)
      << 0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0);
  const super_resolution::ImageData test_image_2(test_image_matrix_2);
}
