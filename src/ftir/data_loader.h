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

 private:
  // The data is stored as independent matrices because the channel count is
  // generally very high.
  std::vector<cv::Mat> data_;
};

}  // namespace ftir
}  // namespace super_resolution

#endif  // SRC_FTIR_DATA_LOADER_H_
