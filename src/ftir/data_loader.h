// Provides an API to load and store FT-IR data sets.

#ifndef SRC_FTIR_DATA_LOADER_H_
#define SRC_FTIR_DATA_LOADER_H_

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {
namespace ftir {

class DataLoader {
 public:
  // Provide a data file name that will be processed.
  explicit DataLoader(const std::string& file_path);

  // Returns the data in pixel form, where each row of the returned matrix
  // represents the values for each band in that pixel. Hence the returned
  // matrix is (num_image_rows_ *num_image_cols_) by num_spectral_bands_.
  //
  // The pixels from the raw image are ordered rows before columns; that is,
  // the first M rows of the returned matrix is the first row (of M columns) in
  // the image, the next M elements is the second row of the image, and so on.
  cv::Mat GetPixelData() const;

 private:
  // The data is stored as independent matrices because the channel count is
  // generally very high.
  std::vector<cv::Mat> data_;
  int num_image_rows_ = 0;
  int num_image_cols_ = 0;
  int num_spectral_bands_ = 0;
};

}  // namespace ftir
}  // namespace super_resolution

#endif  // SRC_FTIR_DATA_LOADER_H_
