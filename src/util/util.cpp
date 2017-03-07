#include "util/util.h"

#include <dirent.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "image/image_data.h"
#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

namespace super_resolution {
namespace util {

// The size of a displayed image for the DisplayImage function if rescale is
// set to true.
constexpr int kDisplaySizePixels = 850;

void InitApp(int argc, char** argv, const std::string& usage_message) {
  google::SetUsageMessage(usage_message);
  google::SetVersionString(kCodeVersion);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // TODO: remove this or put it under a debug guard.
  FLAGS_logtostderr = true;
}

std::string GetRootCodeDirectory() {
#ifdef ROOT_CODE_DIRECTORY
  return std::string(ROOT_CODE_DIRECTORY);
#else
  LOG(WARNING) << "ROOT_CODE_DIRECTORY is not defined. "
               << "Returning local Unix directory ('.')";
  return ".";
#endif
}

std::string GetAbsoluteCodePath(const std::string& relative_path) {
  return GetRootCodeDirectory() + "/" + relative_path;
}

std::vector<std::string> ListFilesInDirectory(const std::string& directory) {
  std::vector<std::string> list_of_files;

  // Try to open the directory. If this fails, return an empty list of files.
  DIR* dir_ptr = opendir(directory.c_str());
  if (dir_ptr == NULL) {
    LOG(ERROR) << "Could not open directory: " << directory;
    return list_of_files;
  }

  // Loop through the contents and only keep non-hidden files.
  struct dirent* dirp;
  while ((dirp = readdir(dir_ptr)) != NULL) {
    // Skip directories.
    if (dirp->d_type != DT_REG) {
      continue;
    }
    const std::string item = std::string(dirp->d_name);
    // Skip hidden files.
    if (item.compare(0, 1, ".") == 0) {
      continue;
    }
    list_of_files.push_back(std::string(dirp->d_name));
  }
  closedir(dir_ptr);

  return list_of_files;
}

std::vector<std::string> SplitString(
    const std::string& whole_string,
    const char delimiter,
    const bool ignore_empty_pieces) {

  std::vector<std::string> parts;
  std::string remaining = whole_string;
  int split_position = remaining.find(delimiter);
  while (split_position >= 0) {
    const std::string part = remaining.substr(0, split_position);
    if (!ignore_empty_pieces || !part.empty()) {
      parts.push_back(part);
    }
    remaining = remaining.substr(split_position + 1);
    split_position = remaining.find(delimiter);
  }
  if (!ignore_empty_pieces || !remaining.empty()) {
    parts.push_back(remaining);
  }

  return parts;
}

// Inspired from:
// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
std::string TrimString(const std::string& untrimmed_string) {
  std::string trimmed_string = untrimmed_string;
  // Trim left.
  trimmed_string.erase(trimmed_string.begin(), std::find_if(
      trimmed_string.begin(),
      trimmed_string.end(),
      std::not1(std::ptr_fun<int, int>(std::isspace))));
  // Trim right.
  trimmed_string.erase(
      std::find_if(
          trimmed_string.rbegin(),
          trimmed_string.rend(),
          std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
      trimmed_string.end());
  return trimmed_string;
}

void DisplayImage(
    const ImageData& image,
    const std::string& window_name,
    const bool rescale) {

  ImageData display_image = image;
  const cv::Size image_size = display_image.GetImageSize();
  if (rescale) {
    const int smaller_dimension = std::min(image_size.width, image_size.height);
    const double scale =
        static_cast<double>(kDisplaySizePixels) /
        static_cast<double>(smaller_dimension);
    if (scale > 1.0) {
      display_image.ResizeImage(scale);
    }
  }

  cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  cv::imshow(window_name, display_image.GetVisualizationImage());
  cv::waitKey(0);
  cv::destroyWindow(window_name);
}

void DisplayImagesSideBySide(
    const std::vector<ImageData>& images,
    const std::string& window_name,
    const bool rescale) {

  CHECK_GT(images.size(), 0) << "Provide at least one image to display.";

  // Concatenate the images side-by-side.
  int width = 0;
  int height = 0;
  for (const ImageData& image : images) {
    const cv::Size image_size = image.GetImageSize();
    width += image_size.width;
    height = std::max(height, image_size.height);
  }

  const int image_type = (images[0].GetNumChannels() < 3) ? CV_8UC1 : CV_8UC3;
  cv::Mat stitched_images(height, width, image_type);
  int x_pos = 0;
  for (const ImageData& image : images) {
    const cv::Size image_size = image.GetImageSize();
    cv::Mat next_region = stitched_images(
        cv::Rect(x_pos, 0, image_size.width, image_size.height));
    image.GetVisualizationImage().copyTo(next_region);
    x_pos += image_size.width;
  }

  // Create an ImageData (force normalization since this is with unsigned
  // values) and display normally.
  ImageData stitched_image_data(stitched_images, NORMALIZE_IMAGE);
  DisplayImage(stitched_image_data, window_name, rescale);
}

int GetPixelIndex(
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col) {

  const int channel_index = channel * (image_size.width * image_size.height);
  return channel_index + (row * image_size.width + col);
}

}  // namespace util
}  // namespace super_resolution
