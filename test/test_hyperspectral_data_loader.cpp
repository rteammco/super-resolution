#include <iostream>
#include <string>

#include "hyperspectral/data_loader.h"

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

  EXPECT_EQ(hs_data_loader.GetNumSpectralBands(), data_size.bands);

  const int num_pixels = data_size.rows * data_size.cols;
  const cv::Mat pixels = hs_data_loader.GetPixelData();
  EXPECT_EQ(pixels.size(), cv::Size(data_size.bands, num_pixels));

  cv::Mat band_0_image = hs_data_loader.GetSpectralBandImage(0);
  EXPECT_EQ(band_0_image.size(), cv::Size(data_size.cols, data_size.rows));

  for (int b = 0; b < data_size.bands; ++b) {
    cv::imshow("test window", hs_data_loader.GetSpectralBandImage(b));
    cv::waitKey(0);
  }

  // Test PCA.
  cv::PCA pca(pixels, cv::Mat(), CV_PCA_DATA_AS_ROW);
  // TODO(richard): remove these prints.
  // std::cout << pca.eigenvectors << std::endl;
  // std::cout << pca.mean << std::endl;
  std::cout << pca.eigenvalues << std::endl;

  cv::Mat indices;
  cv::sortIdx(
      pca.eigenvalues,
      indices,
      CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
  std::cout << indices << std::endl;
}
