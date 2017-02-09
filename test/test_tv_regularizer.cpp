#include <algorithm>
#include <cmath>
#include <vector>

#include "optimization/regularizer.h"
#include "optimization/tv_regularizer.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ContainerEq;
using testing::SizeIs;

// Minimum divisor so we prevent division by zero.
constexpr double kMinDivisor = 0.000001;

// Small test image and expected returned values for this data.
const cv::Size test_image_size(3, 3);
const std::vector<double> test_image_data = {
     0,  0, 1,
     0,  1, 3,
    -3, -1, 0
};
// The x-direction gradients should be:
//
//  | 0 | 1 | 0 |                | 0 | 1 | 0 |
//  | 1 | 2 | 0 |     squared=>  | 1 | 4 | 0 |
//  | 2 | 1 | 0 |                | 4 | 1 | 0 |
//
// and the y-direction gradients should be:
//
//  |  0 |  1 |  2 |             | 0 | 1 | 4 |
//  | -3 | -2 | -3 |  squared=>  | 9 | 4 | 9 |
//  |  0 |  0 |  0 |             | 0 | 0 | 0 |
//
// For the 1-norm, it's a sum of absolute value of the X and Y gradients. For
// the 2-norm version, it's the square root of the squared sums.
const std::vector<double> test_image_expected_residuals_1_norm = {
  (0 + 0), (1 + 1), (0 + 2),
  (1 + 3), (2 + 2), (0 + 3),
  (2 + 0), (1 + 0), (0 + 0)
};
// TODO: probably remove 2-norm tests.
const std::vector<double> test_image_expected_residuals_2_norm = {
  sqrt(0 + 0), sqrt(1 + 1), sqrt(0 + 4),
  sqrt(1 + 9), sqrt(4 + 4), sqrt(0 + 9),
  sqrt(4 + 0), sqrt(1 + 0), sqrt(0 + 0)
};

// Replicates a vector the given number of times to extend it. The new elements
// will be exact copies of the original element sequence.
std::vector<double> ReplicateVector(
    const std::vector<double>& data, const int count) {

  const int num_elements = data.size();
  std::vector<double> replicated = data;
  replicated.resize(num_elements * count);
  for (int i = 1; i < count; ++i) {
    std::copy(data.begin(), data.end(), replicated.begin() + num_elements * i);
  }
  return replicated;
}

// Verifies that the TV regularizer correctly computes the expected residuals
// for an image.
TEST(TotalVariationRegularizer, ApplyToImage) {
  const super_resolution::TotalVariationRegularizer tv_regularizer(
      test_image_size, 3);
  const std::vector<double> input_data = ReplicateVector(test_image_data, 3);
  const std::vector<double> returned_residuals =
      tv_regularizer.ApplyToImage(input_data.data());

  // For testing, copy the expected residuals for each channel.
  std::vector<double> expected_residuals_1_norm = ReplicateVector(
      test_image_expected_residuals_1_norm, 3);
  EXPECT_THAT(returned_residuals, ContainerEq(expected_residuals_1_norm));
}

// Tests the 3D total variation mode of the TotalVariationRegularizer.
TEST(TotalVariationRegularizer, ApplyToImage3d) {
  double input_data[9 * 3] = {
    // Channel 1:
     0,  0,  1,
     0,  1,  3,
    -3, -1,  0,
    // Channel 2:
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
    // Channel 3:
     0, -1,  2,
    -3,  4,  5,
     6,  7, -8
  };

  const std::vector<double> expected_residuals = {
  // Expected TV 3D for channel 1 (diff between ch. 1 and ch 2.):
  // Same 2D total variation, but for the Z-axis:
  //
  //   Channel 1:            Channel 2:
  // |  0 |  0 | 1 |       | 0 | 0 | 0 |
  // |  0 |  1 | 3 |  vs.  | 0 | 0 | 0 |
  // | -3 | -1 | 0 |       | 0 | 0 | 0 |
  //
  // Z diffs are:
  //   0, 0, -1,
  //   0, -1, -3,
  //   3, 1, 0
  //
  // So for channel 1, the same X and Y gradients as for the 2D test image,
  // plus the absolute values of the Z gradients above - (X + Y) + Z:
    (0 + 0) + 0, (1 + 1) + 0, (0 + 2) + 1,
    (1 + 3) + 0, (2 + 2) + 1, (0 + 3) + 3,
    (2 + 0) + 3, (1 + 0) + 1, (0 + 0) + 0,
  //
  // And channel 2 to channel 3, similarly the differences are:
  //   0, -1, 2,
  //   -3, 4, 5,
  //   6, 7, -8
  // (and X and Y are all zero because the image is all zeros):
    (0 + 0) + 0, (0 + 0) + 1, (0 + 0) + 2,
    (0 + 0) + 3, (0 + 0) + 4, (0 + 0) + 5,
    (0 + 0) + 6, (0 + 0) + 7, (0 + 0) + 8,
  //
  // No channel 4, so the 3D TV values of Channel 3 are just the 2D values:
  //   0, -1,  2,        -1, 3, 0
  //  -3,  4,  5,  X =>  7, 1, 0
  //   6,  7, -8         1, -15, 0
  //
  //  Y =>
  //
  //  -3  5  3
  //   9  3 -13
  //   0  0  0
  //
  // So the TV values are:
    (1 + 3) + 0, (3 + 5) + 0, (0 + 3) + 0,
    (7 + 9) + 0, (1 + 3) + 0, (0 + 13) + 0,
    (1 + 0) + 0, (15 + 0) + 0, (0 + 0) + 0
  };

  // Apply the TV Regularizer and verify the returned values.
  super_resolution::TotalVariationRegularizer tv_regularizer(
      test_image_size, 3);  // 3 x 3 x 3 image.
  tv_regularizer.SetUse3dTotalVariation(true);
  const std::vector<double> returned_residuals =
      tv_regularizer.ApplyToImage(input_data);
  EXPECT_THAT(returned_residuals, ContainerEq(expected_residuals));
}

// TODO: test 3D TV once it is implemented.

// Verifies that the derivatives are also being computed correctly.
TEST(TotalVariationRegularizer, ApplyToImageWithDifferentiation) {
  const super_resolution::TotalVariationRegularizer tv_regularizer(
      test_image_size, 1);  // TODO: testing only 1 channel. Test multiple.

  // Given the following indices of the image:
  //
  //  | 0 | 1 | 2 |
  //  | 3 | 4 | 5 |
  //  | 6 | 7 | 8 |
  //
  // we should expect the following relevant partial derivatives - for each
  // pixel, all other pixels w.r.t. that pixel (itself, the one below, and the
  // one to the right).
  //
  //        self    left    above
  // d/d0 = d0/d0 + d1/d0 + d3/d0
  // d/d1 = d1/d1 + d2/d1 + d4/d1
  // d/d2 = d2/d2 +   0   + d5/d2   <= no pixel to the right
  // d/d3 = d3/d3 + d4/d3 + d6/d3
  // d/d4 = d4/d4 + d5/d4 + d7/d4
  // d/d5 = d5/d5 +   0   + d8/d5   <= no pixel to the right
  // d/d6 = d6/d6 + d7/d6 +   0     <= no pixel below
  // d/d7 = d7/d7 + d8/d7 +   0     <= no pixel below
  // d/d8 = d8/d8 +   0   +   0     <= no pixel to the right or below
  //
  // The partials should be (where the numbers represent the pixel values at
  // that image index):
  // TODO: fix this.
  // The others follow similarly, but we won't test those.

  // First, compute the "total variation", which is just the expected residuals
  // but padding zero values with a small delta to avoid division by 0. Total
  // variation is always non-negative.
  std::vector<double> total_variation;
  for (const double residual : test_image_expected_residuals_2_norm) {
    total_variation.push_back(std::max(residual, kMinDivisor));
  }

  // The partial derivatives, calculated from the test image, thus are:
  const double d0d0 =
      ((test_image_data[1] - test_image_data[0]) +
      (test_image_data[3] - test_image_data[0])) / total_variation[0];
  const double d1d1 =
      ((test_image_data[2] - test_image_data[1]) +
      (test_image_data[4] - test_image_data[1])) / total_variation[1];
  const double d1d0 =
      -(test_image_data[1] - test_image_data[0]) / total_variation[0];
  const double d2d2 =
      (test_image_data[5] - test_image_data[2]) / total_variation[2];
  const double d2d1 =
      -(test_image_data[2] - test_image_data[1]) / total_variation[1];
  const double d3d3 =
      ((test_image_data[4] - test_image_data[3]) +
      (test_image_data[6] - test_image_data[3])) / total_variation[3];
  const double d3d0 =
      -(test_image_data[3] - test_image_data[0]) / total_variation[0];
  const double d4d4 =
      ((test_image_data[5] - test_image_data[4]) +
      (test_image_data[7] - test_image_data[4])) / total_variation[4];
  const double d4d3 =
      -(test_image_data[4] - test_image_data[3]) / total_variation[3];
  const double d4d1 =
      -(test_image_data[4] - test_image_data[1]) / total_variation[1];
  const double d8d8 = 0;  // Edge case, no TV defined in corner....
  const double d8d7 =
      -(test_image_data[8] - test_image_data[7]) / total_variation[7];
  const double d8d5 =
      -(test_image_data[8] - test_image_data[5]) / total_variation[5];

  // The expected values for the first four derivatives.
  const double expected_d0 = d0d0;
  const double expected_d1 = d1d1 + d1d0;
  const double expected_d2 = d2d2 + d2d1;
  const double expected_d3 = d3d3 + d3d0;
  const double expected_d4 = d4d4 + d4d3 + d4d1;
  const double expected_d8 = d8d8 + d8d7 + d8d5;

  // Pass identity (1) as weights, so we should expect the pure partial
  // derivatives be returned.
  std::vector<double> gradient_constants(9);
  std::fill(gradient_constants.begin(), gradient_constants.end(), 1.0);

  const auto& returned_residuals_and_gradient_automatic =
      tv_regularizer.ApplyToImageWithDifferentiation(
          test_image_data.data(),
          gradient_constants,
          super_resolution::AUTOMATIC_DIFFERENTIATION);
  const std::vector<double> gradient_automatic =
      returned_residuals_and_gradient_automatic.second;

  // Check correct behavior of returned residuals.
  EXPECT_THAT(
      returned_residuals_and_gradient_automatic.first,
      ContainerEq(test_image_expected_residuals_2_norm));

  EXPECT_THAT(gradient_automatic, SizeIs(9));

  EXPECT_EQ(gradient_automatic[0], expected_d0);
  EXPECT_EQ(gradient_automatic[1], expected_d1);
  EXPECT_EQ(gradient_automatic[2], expected_d2);
  EXPECT_EQ(gradient_automatic[3], expected_d3);
  EXPECT_EQ(gradient_automatic[4], expected_d4);
  EXPECT_EQ(gradient_automatic[8], expected_d8);

  /* Also verify that the gradient is correct with analytical diff. */

  const auto& returned_residuals_and_gradient_analytical =
      tv_regularizer.ApplyToImageWithDifferentiation(
          test_image_data.data(),
          gradient_constants,
          super_resolution::ANALYTICAL_DIFFERENTIATION);
  const std::vector<double> gradient_analytical =
      returned_residuals_and_gradient_analytical.second;

  // Check correct behavior of returned residuals.
  EXPECT_THAT(
      returned_residuals_and_gradient_analytical.first,
      ContainerEq(test_image_expected_residuals_2_norm));

  EXPECT_THAT(gradient_analytical, SizeIs(9));

  EXPECT_EQ(gradient_analytical[0], expected_d0);
  EXPECT_EQ(gradient_analytical[1], expected_d1);
  EXPECT_EQ(gradient_analytical[2], expected_d2);
  EXPECT_EQ(gradient_analytical[3], expected_d3);
  EXPECT_EQ(gradient_analytical[4], expected_d4);
  EXPECT_EQ(gradient_analytical[8], expected_d8);
}
