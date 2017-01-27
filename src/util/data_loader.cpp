#include "util/data_loader.h"

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "hyperspectral/hyperspectral_data_loader.h"
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
static const std::unordered_set<std::string> kSupportedHyperspectralExtensions({
    "txt"
    // TODO: add support for binary extensions.
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
  std::string extension = file_path.substr(file_path.find_last_of(".") + 1);
  std::transform(
      extension.begin(), extension.end(), extension.begin(), tolower);
  if (DoesSetContain(kSupportedImageExtensions, extension)) {
    const cv::Mat image = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);
    return ImageData(image);
  } else if (DoesSetContain(kSupportedHyperspectralExtensions, extension)) {
    hyperspectral::HyperspectralDataLoader hs_data_loader(file_path);
    hs_data_loader.LoadData();
    return hs_data_loader.GetImage();
  }

  // Unsupported/unknown extension, so return empty image.
  LOG(WARNING) << "Could not load image " << file_path
               << ": extension is not recognized.";
  return ImageData();
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
