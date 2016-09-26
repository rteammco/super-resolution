#include <iostream>
#include <string>

#include "ftir/data_loader.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// TODO(richard): make this test data available in the repo and make it
// smaller and more optimal for testing.
static const std::string kTestDataPath = "../data/ftir_test.txt";

TEST(FtirDataLoader, DataLoaderTest) {
  super_resolution::ftir::DataLoader ftir_data_loader(kTestDataPath);
  const cv::Mat pixels = ftir_data_loader.GetPixelData();

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
