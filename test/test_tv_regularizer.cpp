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
}

// This test verifies the behavior of the derivative computation for the TV
// Regularizer.
TEST(TotalVariationRegularizer, GetDerivatives) {
  // TODO: implement tests!
}
