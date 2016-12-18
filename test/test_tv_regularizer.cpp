#include <cmath>
#include <vector>

#include "solvers/tv_regularizer.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ContainerEq;

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

  // Pass identity (1) as weights, so we should expect the pure partial
  // derivatives be returned.
  std::vector<double> partial_const_terms(9);
  std::fill(partial_const_terms.begin(), partial_const_terms.end(), 1.0);

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
  //   d0/d0 = ((0 - 1) + (0 - 3)) / tv(0)
  //   d1/d1 = ((1 - 2) + (1 - 4)) / tv(1)
  //   d1/d0 = -(0 - 1) / tv(0)
  //   d2/d2 = (0 + (5 - 2)) / tv(2)
  //   d2/d1 = -(1 - 2) / tv(1)
  //   d3/d3 = ((3 - 4) + (3 - 6)) / tv(3)
  //   d3/d0 = -(0 - 3) / tv(0)
  //   d4/d4 = ((4 - 5) + (4 - 7)) / tv(4)
  //   d4/d3 = -(3 - 4) / tv(3)
  //   d4/d1 = -(1 - 4) / tv(1)
  // The others follow similarly, we won't test those.
  //
  // The actual derivatives, calculated from the test image, thus are:
  const double d0d0 =
      ((image1_data[0] - image1_data[1]) +
      (image1_data[0] - image1_data[3])) / expected_residuals[0];
  const double d1d1 =
      ((image1_data[1] - image1_data[2]) +
      (image1_data[1] - image1_data[4])) / expected_residuals[1];
  const double d1d0 =
      -(image1_data[0] - image1_data[1]) / expected_residuals[0];

  // TODO: divide by 0, and finish the value calculations.

  const double expected_d0 = d0d0;
  const double expected_d1 = d1d1 + d1d0;
}

// This test verifies the behavior of the derivative computation for the TV
// Regularizer.
TEST(TotalVariationRegularizer, GetDerivatives) {
  // TODO: implement tests!
}
