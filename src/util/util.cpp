#include "util/util.h"

#include <algorithm>
#include <dirent.h>
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

  cv::Mat stitched_images(height, width, CV_8UC3);
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
  ImageData stitched_image_data(stitched_images, true);
  DisplayImage(stitched_image_data, window_name, rescale);
}

}  // namespace util
}  // namespace super_resolution
