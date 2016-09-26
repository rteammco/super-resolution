#include <iostream>
#include <string>

#include "ftir/data_loader.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// TODO(richard): make this test data available in the repo and make it
// smaller and more optimal for testing.
static const std::string kTestDataPath = "../test_data/ftir_test.txt";

TEST(FtirDataLoader, DataLoaderTest) {
  super_resolution::ftir::DataLoader ftir_data_loader(kTestDataPath);

  const int num_spectral_bands = 5;
  EXPECT_EQ(ftir_data_loader.GetNumSpectralBands(), num_spectral_bands);

  const int num_pixels = 128 * 128;
  const cv::Mat pixels = ftir_data_loader.GetPixelData();
  const cv::Size pixels_size = pixels.size();
  EXPECT_EQ(pixels.size(), cv::Size(num_spectral_bands, num_pixels));

  cv::Mat band_0_image = ftir_data_loader.GetSpectralBandImage(0);
  EXPECT_EQ(band_0_image.size(), cv::Size(128, 128));

  cv::imshow("test window", band_0_image);
  cv::waitKey(0);

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
