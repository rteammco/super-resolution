#include <iostream>
#include <string>

#include "hyperspectral/data_loader.h"
#include "image/image_data.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// TODO(richard): make this test data available in the repo and make it
// smaller and more optimal for testing.
static const std::string kTestDataPath = "../test_data/ftir_test.txt";

TEST(HyperspectralDataLoader, DataLoaderTest) {
  const super_resolution::hyperspectral::HyperspectralCubeSize data_size(
      128, 128, 5);
  super_resolution::hyperspectral::DataLoader hs_data_loader(
      kTestDataPath, data_size);
  hs_data_loader.LoadData();
  const super_resolution::ImageData& hs_image = hs_data_loader.GetImage();

  const int num_pixels = data_size.rows * data_size.cols;
  EXPECT_EQ(hs_image.GetNumPixels(), num_pixels);
  EXPECT_EQ(hs_image.GetNumChannels(), data_size.bands);

  for (int b = 0; b < data_size.bands; ++b) {
    cv::imshow("Test Window", hs_image.GetChannelImage(b));
    cv::waitKey(0);
  }
}
