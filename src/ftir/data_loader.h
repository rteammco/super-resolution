// Provides an API to load and store FT-IR data sets.

#ifndef SRC_FTIR_DATA_LOADER_H_
#define SRC_FTIR_DATA_LOADER_H_

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {
namespace ftir {

// Dimensions of the Hyperspectral cube.
struct HyperspectralCubeSize {
  HyperspectralCubeSize(const int rows, const int cols, const int bands)
      : rows(rows), cols(cols), bands(bands) {}

  const int rows;
  const int cols;
  const int bands;
};

class DataLoader {
 public:
  // Provide a data file name that will be processed.
  explicit DataLoader(
      const std::string& file_path, const HyperspectralCubeSize& data_size);

  // Returns the number of bands (channels) in this hyperspectral image.
  int GetNumSpectralBands() const {
    return data_size_.bands;
  }

  // Returns the image for a given spectral band index. The index must be
  // valid: 0 <= band_index < GetNumSpectralBands().
  cv::Mat GetSpectralBandImage(const int band_index) const;

  // Returns the data in pixel form, where each row of the returned matrix
  // represents the values for each band in that pixel. Hence the returned
  // matrix is (num_rows * num_cols) by num_bands.
  //
  // The pixels from the raw image are ordered rows before columns; that is,
  // the first M rows of the returned matrix is the first row (of M columns) in
  // the image, the next M elements is the second row of the image, and so on.
  cv::Mat GetPixelData() const;

 private:
  // The data is stored as independent matrices because the channel count is
  // generally very high.
  std::vector<cv::Mat> data_;

  // The size of the data.
  const HyperspectralCubeSize data_size_;
};

}  // namespace ftir
}  // namespace super_resolution

#endif  // SRC_FTIR_DATA_LOADER_H_
