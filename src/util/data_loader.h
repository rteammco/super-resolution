// Provides data file I/O utility functions, including a LoadImages function
// that can read image files in any supported format from a given data path.

#ifndef SRC_UTIL_DATA_LOADER_H_
#define SRC_UTIL_DATA_LOADER_H_

#include <string>
#include <vector>

#include "image/image_data.h"

namespace super_resolution {
namespace util {

// Returns true if the given path is a directory, and false otherwise. If the
// given path is not valid or cannot be accessed, this will cause an error.
bool IsDirectory(const std::string& path);

// Returns a list of images loaded from the given data_path. If the data_path
// points to a directory, the list will contain images loaded from all files
// in that directory.
//
// The given data_path should be an image file or directory containing
// multiple image files. The file(s) can be one of the following formats:
//   - Standard image file (.jpg, .png, etc.)
//   - Standard video file (.avi, .mpg, etc.)
//   - Hyperspectral data in text format.
//   - TODO: Binary hyperspectral data files.
// Unsupported or invalid files or directories will result in an error.
std::vector<ImageData> LoadImages(const std::string& data_path);

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_DATA_LOADER_H_
