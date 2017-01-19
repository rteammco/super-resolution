#include <algorithm>
#include <cmath>
#include <vector>

#include "optimization/tv_regularizer.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ContainerEq;
using testing::SizeIs;

// Minimum divisor so we prevent division by zero.
constexpr double kMinDivisor = 0.000001;

// Verifies that the TV regularizer correctly computes the expected residuals
// for an image.
TEST(TotalVariationRegularizer, ApplyToImage) {
  const cv::Size image_size(3, 3);
  const super_resolution::TotalVariationRegularizer tv_regularizer(image_size);

  const double image1_data[9] = {
     0,  0, 1,
     0,  1, 3,
    -3, -1, 0
  };
  // The x-direction gradients (not squared) should be:
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
  // So at each pixel, the square roots of the squared sums are expected to be
  // sqrt(y*y + x*x):
  const std::vector<double> expected_residuals = {
    sqrt(0 + 0), sqrt(1 + 1), sqrt(0 + 4),
    sqrt(1 + 9), sqrt(4 + 4), sqrt(0 + 9),
    sqrt(4 + 0), sqrt(1 + 0), sqrt(0 + 0)
  };

  const std::vector<double> returned_residuals =
      tv_regularizer.ApplyToImage(image1_data);
  EXPECT_THAT(returned_residuals, ContainerEq(expected_residuals));

  /* Now verify that the derivatives are being computed correctly. */

  // Given the following indices of the image:
  //
  //  | 0 | 1 | 2 |
  //  | 3 | 4 | 5 |
  //  | 6 | 7 | 8 |
  //
  // we should expect the following relevant partial derivatives (each pixel
  // w.r.t. itself, the pixel to the left, and the pixel above it):
  //
  //        self    left    above
  // d/d0 = d0/d0 +   0   +   0     <= no pixels to the left or above
  // d/d1 = d1/d1 + d1/d0 +   0     <= no pixel above
  // d/d2 = d2/d2 + d2/d1 +   0     <= no pixel above
  // d/d3 = d3/d3 +   0   + d3/d0   <= no pixel to the left
  // d/d4 = d4/d4 + d4/d3 + d4/d1
  // d/d5 = d5/d5 + d5/d4 + d5/d2
  // d/d6 = d6/d6 +   0   + d6/d3   <= no pixel to the left
  // d/d7 = d7/d7 + d7/d6 + d7/d4
  // d/d8 = d8/d8 + d8/d7 + d8/d5
  //
  // The partials should be (where the numbers represent the pixel values at
  // that image index):
  //   d0/d0 = ((1 - 0) + (3 - 0)) / tv(0)
  //   d1/d1 = ((2 - 1) + (4 - 1)) / tv(1)
  //   d1/d0 = -(1 - 0) / tv(0)
  //   d2/d2 = (5 - 2) / tv(2)
  //   d2/d1 = -(2 - 1) / tv(1)
  //   d3/d3 = ((4 - 3) + (6 - 3)) / tv(3)
  //   d3/d0 = -(3 - 0) / tv(0)
  //   d4/d4 = ((4 - 3) + (7 - 4)) / tv(4)
  //   d4/d3 = -(4 - 3) / tv(3)
  //   d4/d1 = -(4 - 1) / tv(1)
  //   d8/d8 = zero (edge case)
  //   d8/d7 = -(8 - 7) / tv(7)
  //   d8/d5 = -(8 - 5) / tv(5)
  // The others follow similarly, but we won't test those.

  // First, compute the "total variation", which is just the expected residuals
  // but padding zero values with a small delta to avoid division by 0. Total
  // variation is always non-negative.
  std::vector<double> total_variation;
  for (const double residual : expected_residuals) {
    total_variation.push_back(std::max(residual, kMinDivisor));
  }

  // The actual derivatives, calculated from the test image, thus are:
  const double d0d0 =
      ((image1_data[1] - image1_data[0]) +
      (image1_data[3] - image1_data[0])) / total_variation[0];
  const double d1d1 =
      ((image1_data[2] - image1_data[1]) +
      (image1_data[4] - image1_data[1])) / total_variation[1];
  const double d1d0 =
      -(image1_data[1] - image1_data[0]) / total_variation[0];
  const double d2d2 =
      (image1_data[5] - image1_data[2]) / total_variation[2];
  const double d2d1 =
      -(image1_data[2] - image1_data[1]) / total_variation[1];
  const double d3d3 =
      ((image1_data[4] - image1_data[3]) +
      (image1_data[6] - image1_data[3])) / total_variation[3];
  const double d3d0 =
      -(image1_data[3] - image1_data[0]) / total_variation[0];
  const double d4d4 =
      ((image1_data[5] - image1_data[4]) +
      (image1_data[7] - image1_data[4])) / total_variation[4];
  const double d4d3 =
      -(image1_data[4] - image1_data[3]) / total_variation[3];
  const double d4d1 =
      -(image1_data[4] - image1_data[1]) / total_variation[1];
  const double d8d8 = 0;  // Edge case, no TV defined in corner....
  const double d8d7 =
      -(image1_data[8] - image1_data[7]) / total_variation[7];
  const double d8d5 =
      -(image1_data[8] - image1_data[5]) / total_variation[5];

  // The expected values for the first four derivatives.
  const double expected_d0 = d0d0;
  const double expected_d1 = d1d1 + d1d0;
  const double expected_d2 = d2d2 + d2d1;
  const double expected_d3 = d3d3 + d3d0;
  const double expected_d4 = d4d4 + d4d3 + d4d1;
  const double expected_d8 = d8d8 + d8d7 + d8d5;

  // Pass identity (1) as weights, so we should expect the pure partial
  // derivatives be returned.
  std::vector<double> partial_const_terms(9);
  std::fill(partial_const_terms.begin(), partial_const_terms.end(), 1.0);

  const std::vector<double> returned_derivatives =
      tv_regularizer.GetDerivatives(image1_data, partial_const_terms);
  EXPECT_THAT(returned_derivatives, SizeIs(9));

  EXPECT_EQ(returned_derivatives[0], expected_d0);
  EXPECT_EQ(returned_derivatives[1], expected_d1);
  EXPECT_EQ(returned_derivatives[2], expected_d2);
  EXPECT_EQ(returned_derivatives[3], expected_d3);
  EXPECT_EQ(returned_derivatives[4], expected_d4);
  EXPECT_EQ(returned_derivatives[8], expected_d8);

  /* Also verify that the derivatives are correct with auto differentiation. */

  const auto& returned_residuals_and_partials =
      tv_regularizer.ApplyToImageWithDifferentiation(image1_data);

  // Check correct behavior of returned residuals.
  EXPECT_THAT(
      returned_residuals_and_partials.first, ContainerEq(expected_residuals));

  EXPECT_EQ(returned_residuals_and_partials.second[0], expected_d0);
  EXPECT_EQ(returned_residuals_and_partials.second[1], expected_d1);
  EXPECT_EQ(returned_residuals_and_partials.second[2], expected_d2);
  EXPECT_EQ(returned_residuals_and_partials.second[3], expected_d3);
  EXPECT_EQ(returned_residuals_and_partials.second[4], expected_d4);
  EXPECT_EQ(returned_residuals_and_partials.second[8], expected_d8);
}
