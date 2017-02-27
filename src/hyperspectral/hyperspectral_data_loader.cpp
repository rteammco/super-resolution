#include "hyperspectral/hyperspectral_data_loader.h"

#include <fstream>
#include <limits>
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

// TODO: Currently, this only loads a single hyperspectral image from text
// format exported from Matlab using dlmwrite.
void HyperspectralDataLoader::LoadData() {
  std::ifstream fin(file_path_);
  CHECK(fin.is_open()) << "Could not open file " << file_path_;

  // The files are organized as follows:
  //
  // First line contains 3 numbers: width, height, and number of bands.
  // Each other line is a row of the data cube.
  //   On each line, nBands spectral values for first col, then for next col...
  //
  // Example: for a 3 x 2 x 4 image (rows, cols, bands) where first digit is
  // row#, then col#, then band# (e.g. 123 is row 1, col 2, band 3):
  //   000 001 002 003 010 011 012 013 020 021 022 023
  //   100 101 102 103 110 111 112 113 120 121 122 123
  //   200 201 202 203 210 211 212 213 220 221 222 223
  //
  // All values are space-delimited.

  // Read first line to get size of the data.
  int num_rows, num_cols, num_bands;
  fin >> num_rows >> num_cols >> num_bands;
  CHECK_GT(num_rows, 0) << "Number of rows must be positive.";
  CHECK_GT(num_cols, 0) << "Number of columns must be positive.";
  CHECK_GT(num_bands, 0) << "Number of spectral bands must be positive.";
  // Skip to the end of the first line and be done with it.
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  // Read in the data. data_cube is a 3D vector organized as [row][col][band].
  std::vector<std::vector<std::vector<double>>> data_cube;
  data_cube.resize(num_rows);
  for (int row = 0; row < num_rows; ++row) {
    std::string line;
    std::getline(fin, line);
    data_cube[row].resize(num_cols);

    std::istringstream token_stream(line);
    std::string token;
    int token_number = 0;
    while (std::getline(token_stream, token, kDataDelimiter)) {
      const double value = std::stof(token);
      const int col = token_number % num_cols;
      data_cube[row][col].push_back(value);
      token_number++;
    }
  }

  // Convert the raw data to matrix form.
  for (int band = 0; band < num_bands; ++band) {
    cv::Mat band_matrix = cv::Mat::zeros(
        num_rows, num_cols, util::kOpenCvMatrixType);
    for (int row = 0; row < num_rows; ++row) {
      for (int col = 0; col < num_cols; ++col) {
        band_matrix.at<double>(row, col) = data_cube[row][col][band];
      }
    }
    hyperspectral_image_.AddChannel(band_matrix, DO_NOT_NORMALIZE_IMAGE);
  }
  CHECK_EQ(hyperspectral_image_.GetNumChannels(), num_bands)
      << "Number of spectral bands does not match the given size.";

  LOG(INFO) << "Successfully loaded a ("
            << num_rows << " rows) x (" << num_cols << " cols) "
            << "image with " << num_bands << " spectral bands.";
}

ImageData HyperspectralDataLoader::GetImage() const {
  CHECK_GT(hyperspectral_image_.GetNumChannels(), 0)
      << "The hyperspectral image is empty. "
      << "Make sure to call LoadData() first.";
  return hyperspectral_image_;
}

void HyperspectralDataLoader::WriteImage(const ImageData& image) const {
  const cv::Size image_size = image.GetImageSize();
  const int num_channels = image.GetNumChannels();
  const int num_rows = image_size.height;
  const int num_cols = image_size.width;
  CHECK_GT(num_rows, 0) << "Number of rows in the image must be positive.";
  CHECK_GT(num_cols, 0) << "Number of columns in the image must be positive.";
  CHECK_GT(num_channels, 0) << "Number of bands in the image must be positive.";

  std::ofstream fout(file_path_);
  CHECK(fout.is_open()) << "Could not open file " << file_path_;

  // Write the meta data (number of rows, cols, and bands).
  fout << num_rows << " " << num_cols << " " << num_channels << "\n";

  // Write the data line by line.
  const int num_pixels = image.GetNumPixels();
  for (int row = 0; row < num_rows; ++row) {
    for (int col = 0; col < num_cols; ++col) {
      for (int channel = 0; channel < num_channels; ++channel) {
        fout << image.GetPixelValue(channel, row, col) << " ";
      }
    }
    fout << "\n";
  }

  fout.close();
}

}  // namespace hyperspectral
}  // namespace super_resolution
