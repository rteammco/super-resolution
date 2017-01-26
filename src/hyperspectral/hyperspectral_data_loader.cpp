#include "hyperspectral/hyperspectral_data_loader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "image/image_data.h"
#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace hyperspectral {

constexpr char kDataDelimiter = ',';

HyperspectralDataLoader::HyperspectralDataLoader(
    const std::string& file_path,
    const HyperspectralCubeSize& data_size)
    : file_path_(file_path), data_size_(data_size) {

  // Sanity check the data size.
  CHECK_GT(data_size_.rows, 0) << "Number of rows must be greater than 0.";
  CHECK_GT(data_size_.cols, 0) << "Number of cols must be greater than 0.";
  CHECK_GT(data_size_.bands, 0) << "Number of bands must be greater than 0.";
}

// TODO: Currently, this only loads a single hyperspectral image from text
// format exported from Matlab using dlmwrite.
void HyperspectralDataLoader::LoadData() {
  std::ifstream fin(file_path_);
  CHECK(fin.is_open()) << "Could not open file " << file_path_;

  LOG(INFO) << "Loading data from file " << file_path_ << "...";

  // The files are organized as follows:
  //
  // Each line is a row.
  //   On each line, nBands spectral values for first col, then for next col...
  //
  // Example: for a 3 x 2 x 4 image (rows, cols, bands) where first digit is
  // row#, then col#, then band# (e.g. 123 is row 1, col 2, band 3):
  //   000 001 002 003 010 011 012 013 020 021 022 023
  //   100 101 102 103 110 111 112 113 120 121 122 123
  //   200 201 202 203 210 211 212 213 220 221 222 223

  // data_cube is a 3D vector organized as [row][col][band].
  std::vector<std::vector<std::vector<double>>> data_cube;
  data_cube.resize(data_size_.rows);
  for (int row = 0; row < data_size_.rows; ++row) {
    std::string line;
    std::getline(fin, line);
    data_cube[row].resize(data_size_.cols);

    std::istringstream token_stream(line);
    std::string token;
    int token_number = 0;
    while (std::getline(token_stream, token, kDataDelimiter)) {
      const double value = std::stof(token);
      const int col = token_number % data_size_.cols;
      data_cube[row][col].push_back(value);
      token_number++;
    }
  }

  // Convert the raw data to matrix form.
  for (int band = 0; band < data_size_.bands; ++band) {
    cv::Mat band_matrix = cv::Mat::zeros(
        data_size_.rows, data_size_.cols, util::kOpenCvMatrixType);
    for (int row = 0; row < data_size_.rows; ++row) {
      for (int col = 0; col < data_size_.cols; ++col) {
        band_matrix.at<double>(row, col) = data_cube[row][col][band];
      }
    }
    hyperspectral_image_.AddChannel(band_matrix);
  }
  CHECK_EQ(hyperspectral_image_.GetNumChannels(), data_size_.bands)
      << "Number of spectral bands does not match the given size.";

  LOG(INFO) << "Done: successfully loaded a ("
            << data_size_.rows << " rows) x (" << data_size_.cols << " cols) "
            << "image with " << data_size_.bands << " spectral bands.";
}

const ImageData& HyperspectralDataLoader::GetImage() const {
  CHECK_GT(hyperspectral_image_.GetNumChannels(), 0)
      << "The hyperspectral image is empty. "
      << "Make sure to call LoadData() first.";
  return hyperspectral_image_;
}

}  // namespace hyperspectral
}  // namespace super_resolution
