#include "ftir/data_loader.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include "glog/logging.h"

namespace super_resolution {
namespace ftir {

constexpr int kFtirImageSize = 128;
constexpr char kDataDelimiter = ',';

DataLoader::DataLoader(const std::string& file_path) {
  // Sanity check the constants, just in case they get changed.
  CHECK_GT(kFtirImageSize, 0);

  std::ifstream fin(file_path);
  CHECK(fin.is_open()) << "Could not open file " << file_path;

  LOG(INFO) << "Loaded data from file " << file_path;

  data_cube_.resize(kFtirImageSize);
  for (int row = 0; row < kFtirImageSize; ++row) {
    std::string line;
    std::getline(fin, line);
    data_cube_[row].resize(kFtirImageSize);

    std::istringstream token_stream(line);
    std::string token;
    int token_number = 0;
    while (std::getline(token_stream, token, kDataDelimiter)) {
      const float value = std::stof(token);
      const int col = token_number % kFtirImageSize;
      data_cube_[row][col].push_back(value);
      token_number++;
    }
  }

  // Note that we resized the vector and verified that the dimension is bigger
  // than 0, so it is safe to access the element at position [0][0].
  const int num_bands = data_cube_[0][0].size();
  LOG(INFO) << "Done: successfully loaded a "
            << kFtirImageSize << " x " << kFtirImageSize << " image with "
            << num_bands << " spectral bands.";
}

}  // namespace ftir
}  // namespace super_resolution
