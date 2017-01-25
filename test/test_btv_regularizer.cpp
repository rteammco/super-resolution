#include "optimization/btv_regularizer.h"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::SizeIs;

// Small test image and expected returned values for this data.
const cv::Size test_image_size(5, 5);
const double test_image_data[25] = {
   0,  0, 1, 2,  1,
   0,  1, 3, 2,  3,
   5,  4, 3, -2, 1,
   4,  6, 9, 3,  0,
  -3, -1, 0, 6,  0
};

TEST(BilateralTotalVariationRegularizer, ApplyToImage) {
  const super_resolution::BilateralTotalVariationRegularizer btv_regularizer(
      test_image_size, 1, 2, 0.5);
  // If scale_range_ is 2 and spatial_decay is 0.5, then:
  // for pixel at (0, 0):
  //   BTV = pow(0.5, 0) * |0 - 0| = 0                    <- compare to itself
  //       + pow(0.5, 1) * |0 - 0| = 0                    <- one to the right
  //       + pow(0.5, 2) * |0 - 1| = 0.25 * 1 = 0.25      <- two to the right
  //       + pow(0.5, 1) * |0 - 0| = 0                    <- one down
  //       + pow(0.5, 2) * |0 - 1| = 0.25                 <- one down, one right
  //       + pow(0.5, 3) * |0 - 3| = 0.125 * 3 = 0.375    <- one down, two right
  //       + pow(0.5, 2) * |0 - 5| = 0.25 * 5 = 1.25      <- two down
  //       + pow(0.5, 3) * |0 - 4| = 0.125 * 4 = 0.5      <- two down, one right
  //       + pow(0.5, 4) * |0 - 3| = 0.0625 * 3 = 0.1875  <- two down, two right
  //  = 2.8125
  const std::vector<double> result_1 =
      btv_regularizer.ApplyToImage(test_image_data);
  EXPECT_THAT(result_1, SizeIs(25));
  EXPECT_DOUBLE_EQ(result_1[0], 2.8125);

  // For the last pixel, it should be zero, since there is nothing to the left
  // or bottom of it.
  EXPECT_DOUBLE_EQ(result_1[24], 0.0);

  /* Run another test with different params and two image channels. */

  // Duplicate the image data into two channels.
  double test_image_data_two_channel[50];
  std::copy(test_image_data, test_image_data + 25, test_image_data_two_channel);
  std::copy(
      test_image_data, test_image_data + 25, test_image_data_two_channel + 25);

  // Set up BTV Regularizer with a range of only 1 and decay of 0.25.
  const super_resolution::BilateralTotalVariationRegularizer btv_regularizer_2(
      test_image_size, 2, 1, 0.25);
  // If scale_range_ is 1 and spatial_decay is 0.25, then:
  // for pixel at (1, 2) - index 7 - is:
  //   BTV = pow(0.25, 0) * |3 - 3|  = 0                   <- compare to itself
  //       + pow(0.25, 1) * |3 - 2|  = 0.25 * 1 = 0.25     <- one left
  //       + pow(0.25, 1) * |3 - 3|  = 0.25 * 0 = 0        <- one down
  //       + pow(0.25, 2) * |3 - -2| = 0.0625 * 5 = 0.3125  <- one down, one left
  //   = 0.5625
  // Since both channels are identical, expect the same value in both.
  const std::vector<double> result_2 =
      btv_regularizer_2.ApplyToImage(test_image_data_two_channel);
  EXPECT_THAT(result_2, SizeIs(50));
  EXPECT_DOUBLE_EQ(result_2[7], 0.5625);
  EXPECT_DOUBLE_EQ(result_2[25 + 7], 0.5625);

  // Last pixel should once again have BTV of zero.
  EXPECT_DOUBLE_EQ(result_2[24], 0.0);
  EXPECT_DOUBLE_EQ(result_2[49], 0.0);
}

TEST(BilateralTotalVariationRegularizer, ApplyToImageWithDifferentiation) {
  // Run the same test as the first test in ApplyToImage, but with
  // differentiation. Use all 0.5 for gradient constants.
  std::vector<double> gradient_constants(25);
  std::fill(gradient_constants.begin(), gradient_constants.end(), 0.5);
  const super_resolution::BilateralTotalVariationRegularizer btv_regularizer(
      test_image_size, 1, 2, 0.5);

  const auto& residuals_and_gradient =
      btv_regularizer.ApplyToImageWithDifferentiation(
          test_image_data, gradient_constants);
  const std::vector<double> residuals = residuals_and_gradient.first;
  const std::vector<double> gradient = residuals_and_gradient.second;

  // Expect same residuals as before.
  EXPECT_THAT(residuals, SizeIs(25));
  EXPECT_DOUBLE_EQ(residuals[0], 2.8125);
  EXPECT_DOUBLE_EQ(residuals[24], 0.0);

  // TODO: check the gradient for accuracy.
}
