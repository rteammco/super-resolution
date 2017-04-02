#include "util/visualization.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "image/image_data.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace util {
namespace {

// The size of a displayed image for the DisplayImage function if rescale is
// set to true.
constexpr int kDisplayWidthPixels = 1250;
constexpr int kDisplayHeightPixels = 850;

}  // namespace

void DisplayImage(
    const ImageData& image,
    const std::string& window_name,
    const bool rescale) {

  ImageData display_image = image;
  const cv::Size image_size = display_image.GetImageSize();
  if (rescale) {
    const double scale_x =
        static_cast<double>(kDisplayWidthPixels) /
        static_cast<double>(image_size.width);
    const double scale_y =
        static_cast<double>(kDisplayHeightPixels) /
        static_cast<double>(image_size.height);
    const double scale = std::min(scale_x, scale_y);
    display_image.ResizeImage(scale);
  }

  cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  cv::imshow(window_name, display_image.GetVisualizationImage());
  std::cout << "Displaying image. Press any key to continue." << std::endl;
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

}  // namespace util
}  // namespace super_resolution