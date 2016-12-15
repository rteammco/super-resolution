#include "ftir/data_loader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "util/util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace ftir {

constexpr char kDataDelimiter = ',';

// TODO: Currently, this only loads a single FT-IR image from text format
// exported from Matlab using dlmwrite.
DataLoader::DataLoader(
    const std::string& file_path, const HyperspectralCubeSize& data_size)
    : data_size_(data_size) {

  // Sanity check the data size.
  CHECK_GT(data_size_.rows, 0) << "Number of rows must be greater than 0.";
  CHECK_GT(data_size_.cols, 0) << "Number of cols must be greater than 0.";
  CHECK_GT(data_size_.bands, 0) << "Number of bands must be greater than 0.";

  std::ifstream fin(file_path);
  CHECK(fin.is_open()) << "Could not open file " << file_path;

  LOG(INFO) << "Loading data from file " << file_path << "...";

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

  std::vector<cv::Mat> channels;
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

  // Convert to matrix form.
  for (int band = 0; band < data_size_.bands; ++band) {
    cv::Mat band_matrix = cv::Mat::zeros(
        data_size_.rows, data_size_.cols, util::kOpenCvMatrixType);
    for (int row = 0; row < data_size_.rows; ++row) {
      for (int col = 0; col < data_size_.cols; ++col) {
        band_matrix.at<double>(row, col) = data_cube[row][col][band];
      }
    }
    data_.push_back(band_matrix);
  }
  CHECK_EQ(data_.size(), data_size_.bands)
      << "Number of spectral bands does not match the given size.";

  LOG(INFO) << "Done: successfully loaded a ("
            << data_size_.rows << " rows) x (" << data_size_.cols << " cols) "
            << "image with " << data_size_.bands << " spectral bands.";
}

cv::Mat DataLoader::GetSpectralBandImage(const int band_index) const {
  CHECK_GE(band_index, 0) << "Band index must be greater than 0.";
  CHECK_LT(band_index, data_size_.bands)
      << "Band index must be less than the number of bands: "
      << data_size_.bands;

  return data_[band_index];
}

cv::Mat DataLoader::GetPixelData() const {
  const int num_pixels = data_size_.rows * data_size_.cols;
  cv::Mat pixels(num_pixels, data_size_.bands, util::kOpenCvMatrixType);

  // Flatten the matrix of each band into a single column vector (O(1)
  // operation) and then copy it to the pixel matrix.
  for (int band = 0; band < data_size_.bands; ++band) {
    const cv::Mat band_matrix = data_[band];
    const cv::Mat band_vector = band_matrix.reshape(1, num_pixels);
    band_vector.copyTo(pixels.col(band));
  }

  return pixels;
}

}  // namespace ftir
}  // namespace super_resolution
