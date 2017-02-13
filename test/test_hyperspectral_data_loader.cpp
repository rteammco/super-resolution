#include <iostream>
#include <string>

#include "hyperspectral/hyperspectral_data_loader.h"
#include "image/image_data.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// Data is 128 x 128 x 5 (width, height, number of bands).
static const std::string kTestDataPath = "../test_data/ftir_test.txt";

TEST(HyperspectralDataLoader, DataLoaderTest) {
  super_resolution::hyperspectral::HyperspectralDataLoader hs_data_loader(
      kTestDataPath);
  hs_data_loader.LoadData();
  const super_resolution::ImageData& hs_image = hs_data_loader.GetImage();

  const int num_pixels = 128 * 128;  // Test image is 128 x 128 x 5.
  EXPECT_EQ(hs_image.GetNumPixels(), num_pixels);
  EXPECT_EQ(hs_image.GetNumChannels(), 5);

//  for (int b = 0; b < 5; ++b) {
//    cv::imshow("Test Window", hs_image.GetChannelImage(b));
//    cv::waitKey(0);
//  }
}
