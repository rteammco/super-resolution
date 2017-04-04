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
      test_image_size);
  const std::vector<double> input_data = ReplicateVector(test_image_data, 3);
  const std::vector<double> returned_residuals =
      tv_regularizer.ApplyToImage(input_data.data(), 3);  // 3 channels

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
      test_image_size);  // 3 x 3 image.
  tv_regularizer.SetUse3dTotalVariation(true);
  const std::vector<double> returned_residuals =
      tv_regularizer.ApplyToImage(input_data, 3);  // 3 channels
  EXPECT_THAT(returned_residuals, ContainerEq(expected_residuals));
}

// TODO: test 3D TV once it is implemented.

// Verifies that the derivatives are also being computed correctly.
TEST(TotalVariationRegularizer, ApplyToImageWithDifferentiation) {
  const super_resolution::TotalVariationRegularizer tv_regularizer(
      test_image_size);

  // Keep all constants neutral for this test.
  std::vector<double> gradient_constants(9);
  std::fill(gradient_constants.begin(), gradient_constants.end(), 1.0);

  const auto& residuals_and_gradient =
      tv_regularizer.ApplyToImageWithDifferentiation(
          test_image_data.data(), gradient_constants, 1);  // 1 channel
  const std::vector<double>& residuals = residuals_and_gradient.first;
  const std::vector<double>& gradient = residuals_and_gradient.second;
  EXPECT_THAT(residuals, ContainerEq(test_image_expected_residuals_1_norm));

  // Compute the true approximate gradient using finite differences (numerical
  // differentiation).
  const double finite_difference = 1e-6;
  const double gradient_error_tolerance = 0.0001;
  for (int i = 0; i < 9; ++i) {
    // Get residual with positive finite difference (+d).
    std::vector<double> pos_diff_image_data = test_image_data;
    pos_diff_image_data[i] += finite_difference;
    const std::vector<double> pos_diff_residuals =
        tv_regularizer.ApplyToImage(pos_diff_image_data.data(), 1);
    double pos_diff_residual = 0.0;
    for (const double residual : pos_diff_residuals) {
      // The residuals are squared because the analytical differentiation is
      // implemented assuming the 2-norm.
      pos_diff_residual += (residual * residual);
    }

    // Get residual with negative finite difference (-d).
    std::vector<double> neg_diff_image_data = test_image_data;
    neg_diff_image_data[i] -= finite_difference;
    const std::vector<double> neg_diff_residuals =
        tv_regularizer.ApplyToImage(neg_diff_image_data.data(), 1);
    double neg_diff_residual = 0.0;
    for (const double residual : neg_diff_residuals) {
      neg_diff_residual += (residual * residual);
    }

    // Compute the partial w.r.t. variable i:
    const double numerical_gradient_at_i =
        (pos_diff_residual - neg_diff_residual) /
        (2 * finite_difference);
    EXPECT_NEAR(numerical_gradient_at_i, gradient[i], gradient_error_tolerance);
  }
}
