#include "ftir/data_loader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace ftir {

constexpr int kFtirImageSize = 128;
constexpr char kDataDelimiter = ',';

// TODO(richard): Currently, this only loads a single FT-IR image from text
// format exported from Matlab using dlmwrite.
DataLoader::DataLoader(const std::string& file_path) {
  // Sanity check the constants, just in case they get changed.
  CHECK_GT(kFtirImageSize, 0);

  std::ifstream fin(file_path);
  CHECK(fin.is_open()) << "Could not open file " << file_path;

  LOG(INFO) << "Loading data from file " << file_path << "...";

  std::vector<cv::Mat> channels;
  std::vector<std::vector<std::vector<double>>> data_cube;
  data_cube.resize(kFtirImageSize);
  for (int row = 0; row < kFtirImageSize; ++row) {
    std::string line;
    std::getline(fin, line);
    data_cube[row].resize(kFtirImageSize);

    std::istringstream token_stream(line);
    std::string token;
    int token_number = 0;
    while (std::getline(token_stream, token, kDataDelimiter)) {
      const double value = std::stof(token);
      const int col = token_number % kFtirImageSize;
      data_cube[row][col].push_back(value);
      token_number++;
    }
  }

  // Note that we resized the vector and verified that the dimension is bigger
  // than 0, so it is safe to access the element at position [0][0].
  const int num_bands = data_cube[0][0].size();

  // Convert to matrix form.
  for (int b = 0; b < num_bands; ++b) {
    cv::Mat band_matrix =
        cv::Mat::zeros(kFtirImageSize, kFtirImageSize, CV_64F);
    for (int row = 0; row < kFtirImageSize; ++row) {
      for (int col = 0; col < kFtirImageSize; ++col) {
        band_matrix.at<double>(row, col) = data_cube[row][col][b];
      }
    }
    data_.push_back(band_matrix);
  }

  LOG(INFO) << "Done: successfully loaded a "
            << kFtirImageSize << " x " << kFtirImageSize << " image with "
            << num_bands << " spectral bands.";
}

}  // namespace ftir
}  // namespace super_resolution
