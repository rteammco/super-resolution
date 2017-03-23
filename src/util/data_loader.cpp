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
namespace {

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

// Returns true if the given set contains the given value.
bool DoesSetContain(
    const std::unordered_set<std::string> set, const std::string& key) {

  return set.find(key) != set.end();
}

}  // namespace

bool IsDirectory(const std::string& path) {
  struct stat path_stat;
  CHECK(stat(path.c_str(), &path_stat) == 0)
      << "The given file or directory path '" << path << "' cannot be opened.";
  return S_ISDIR(path_stat.st_mode);
}

bool IsSupportedImageExtension(const std::string& extension) {
  return DoesSetContain(kSupportedImageExtensions, extension);
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

ImageData LoadImage(const std::string& file_path) {
  std::string extension = file_path.substr(file_path.find_last_of(".") + 1);
  std::transform(
      extension.begin(), extension.end(), extension.begin(), tolower);
  // If the extension is a standard image type, try reading it.
  if (IsSupportedImageExtension(extension)) {
    const cv::Mat image = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    CHECK(!image.empty()) << "Could not load image '" << file_path << "'.";
    return ImageData(image);
  } else {
    // Otherwise, try loading it as a hyperspectral image (assuming the given
    // path was a configuration file).
    HyperspectralDataLoader hs_data_loader(file_path);
    hs_data_loader.LoadImageFromENVIFile();
    return hs_data_loader.GetImage();
  }
}

void SaveImage(const ImageData& image, const std::string& data_path) {
  const int num_channels = image.GetNumChannels();
  if (num_channels == 1 || num_channels == 3) {
    // Monochrome or RGB images are put back together and saved with OpenCV.
    cv::imwrite(data_path, image.GetVisualizationImage());
  } else if (num_channels > 0) {
    // 2 or 4+ channel images are saved as hyperspectral images.
    const HyperspectralDataLoader hs_data_loader(data_path);
    HSIBinaryDataFormat binary_data_format;  // Default format.
    hs_data_loader.SaveImage(image, binary_data_format);
  } else {
    // Can't save an empty image.
    LOG(WARNING) << "Cannot save an empty image. Nothing was saved.";
  }
}

}  // namespace util
}  // namespace super_resolution
