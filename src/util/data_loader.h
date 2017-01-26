// Provides a DataLoader utility class that can read image files in any
// supported format from a given data path.

#ifndef SRC_UTIL_DATA_LOADER_H_
#define SRC_UTIL_DATA_LOADER_H_

#include <vector>

#include "image/image_data.h"

namespace super_resolution {
namespace util {

class DataLoader {
 public:
  // The given data_path should be an image file or directory containing
  // multiple image files. The file(s) can be one of the following formats:
  //   - Standard image file (.jpg, .png, etc.)
  //   - Standard video file (.avi, .mpg, etc.)
  //   - Hyperspectral data in text format.
  //   - TODO: Binary hyperspectral data files.
  // Unsupported files or directories containing unsupported files will result
  // in an error.
  explicit DataLoader(const std::string& data_path);

  // Returns a list of images loaded from the given data_path. If the data_path
  // points to a directory, the list will contain images loaded from all files
  // in that directory.
  std::vector<ImageData> LoadImages() const;

 private:
  const std::string data_path_;
};

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_DATA_LOADER_H_
