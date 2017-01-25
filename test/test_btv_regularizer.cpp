#include "optimization/btv_regularizer.h"

#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

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
  super_resolution::BilateralTotalVariationRegularizer btv_regularizer(
      test_image_size, 1, 2, 0.5);  // TODO: testing only 1 channel. Test multiple.
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
  EXPECT_DOUBLE_EQ(result_1[0], 2.8125);

  // For the last pixel, it should be zero, since there is nothing to the left
  // or bottom of it.
  EXPECT_DOUBLE_EQ(result_1[24], 0.0);
}

TEST(BilateralTotalVariationRegularizer, ApplyToImageWithDifferentiation) {
  // TODO: implement test.
}
