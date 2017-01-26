#include "util/data_loader.h"

#include <algorithm>
#include <dirent.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#include "image/image_data.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace util {

// Supported standard image file extensions (color or grayscale).
static const std::unordered_set<std::string> kSupportedImageExtensions({
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "png",
    "pbm",
    "pgm",
    "ppm"
});

// Supported text-based hyperspectral data file extensions.
static const std::unordered_set<std::string> kSupportedTextHSExtensions({
    "txt"
});

// Supported binary hyperspectral data file extensions.
static const std::unordered_set<std::string> kSupportedBinaryHSExtensions({
    // TODO: add support.
});

// Returns true if the given set contains the given value.
bool DoesSetContain(
    const std::unordered_set<std::string> set, const std::string& key) {

  return set.find(key) != set.end();
}

// Returns a single image loaded from the given file path. The file can be any
// supported type that contains image data.
// This is not a publically accessible function.
ImageData LoadImage(const std::string& file_path) {
  LOG(INFO) << "FILE: " << file_path;
  ImageData image_data;
  std::string extension = file_path.substr(file_path.find_last_of(".") + 1);
  std::transform(
      extension.begin(), extension.end(), extension.begin(), tolower);
  if (DoesSetContain(kSupportedImageExtensions, extension)) {
    const cv::Mat image = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);
    image_data.AddChannel(image);
  } else if (DoesSetContain(kSupportedTextHSExtensions, extension)) {
    // TODO: the HyperspectralDataLoader needs the data size to be known...
  } else if (DoesSetContain(kSupportedBinaryHSExtensions, extension)) {
    // TODO: support binary hyperspectral data as well.
  } else {
    LOG(WARNING) << "Could not load image " << file_path
                 << ": extension is not recognized.";
  }
  return image_data;
}

bool IsDirectory(const std::string& path) {
  struct stat path_stat;
  CHECK(stat(path.c_str(), &path_stat) == 0)
      << "The given file or directory path '" << path << "' cannot be opened.";
  return S_ISDIR(path_stat.st_mode);
}

std::vector<ImageData> LoadImages(const std::string& data_path) {
  std::vector<ImageData> images;
  if (IsDirectory(data_path)) {
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(data_path.c_str())) != NULL) {
      while ((ent = readdir(dir)) != NULL) {
        const std::string file_name(ent->d_name);
        const std::string file_path = data_path + "/" + file_name;
        if (!IsDirectory(file_path)) {
          images.push_back(LoadImage(file_path));
        }
      }
      closedir(dir);
    }
  } else {
    images.push_back(LoadImage(data_path));
  }
  return images;
}

}  // namespace util
}  // namespace super_resolution
