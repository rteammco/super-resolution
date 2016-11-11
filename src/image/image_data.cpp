#include "image/image_data.h"

#include <utility>
#include <vector>

#include "util/util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// Default constructor.
ImageData::ImageData() {
  image_size_ = cv::Size(0, 0);
}

// Copy constructor.
ImageData::ImageData(const ImageData& other) : image_size_(other.image_size_) {
  for (const cv::Mat& channel_image : other.channels_) {
    channels_.push_back(channel_image.clone());
  }
}

// Constructor from OpenCV image.
ImageData::ImageData(const cv::Mat& image) {
  image_size_ = image.size();
  cv::split(image, channels_);  // cv::split copies the data.

  // Make sure the channels are all scaled between 0 and 1.
  for (int i = 0; i < channels_.size(); ++i) {
    // TODO: this only works if the given pixel values are 0 to 255.
    channels_[i].convertTo(channels_[i], util::kOpenCvMatrixType, 1.0 / 255.0);
  }
}

void ImageData::AddChannel(const cv::Mat& channel_image) {
  if (channels_.empty()) {
    image_size_ = channel_image.size();
  } else {
    // Check size and type first if other channels already exist.
    CHECK(channel_image.size() == image_size_)
        << "Channel size did not match the expected size: "
        << channels_[0].size() << " size expected, "
        << channel_image.size() << " size given.";
  }

  // TODO: this only works if the given pixel values are 0 to 255.
  cv::Mat converted_image = channel_image.clone();
  converted_image.convertTo(
      converted_image, util::kOpenCvMatrixType, 1.0 / 255.0);
  channels_.push_back(converted_image);
}

void ImageData::ResizeImage(
    const cv::Size& new_size, const int interpolation_method) {

  // Undefined behavior if image is empty.
  CHECK(!channels_.empty()) << "Cannot resize an empty image.";
  CHECK_GT(new_size.width, 0) << "Images must have a positive width.";
  CHECK_GT(new_size.height, 0) << "Images must have a positive height.";

  const int num_image_channels = channels_.size();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat scaled_image;
    cv::resize(
        channels_[i],   // Source image.
        scaled_image,   // Dest image.
        new_size,       // Desired image size.
        0,              // Set x, y scale to 0 to use the given Size instead.
        0,
        interpolation_method);
    channels_[i] = scaled_image;
  }
  image_size_ = new_size;
}

void ImageData::ResizeImage(
    const double scale_factor, const int interpolation_method) {

  // Undefined behavior if image is empty.
  CHECK(!channels_.empty()) << "Cannot resize an empty image.";
  CHECK_GT(scale_factor, 0) << "Scale factor must be larger than 0.";
  cv::Size new_size(
      static_cast<int>(image_size_.width * scale_factor),
      static_cast<int>(image_size_.height * scale_factor));
  ResizeImage(new_size, interpolation_method);
}

int ImageData::GetNumPixels() const {
  return image_size_.width * image_size_.height;  // (0, 0) if image is empty.
}

cv::Mat ImageData::GetChannelImage(const int index) const {
  CHECK_GE(index, 0) << "Channel index must be at least 0.";
  CHECK_LT(index, channels_.size()) << "Channel index out of bounds.";
  return channels_[index];
}

double ImageData::GetPixelValue(
    const int channel_index, const int pixel_index) const {

  CHECK_GE(channel_index, 0) << "Channel index must be at least 0.";
  CHECK_LT(channel_index, channels_.size()) << "Channel index out of bounds.";

  const std::pair<int, int> image_coordinates =
      GetPixelCoordinatesFromIndex(pixel_index);  // Checks pixel index range.
  return channels_[channel_index].at<double>(
      image_coordinates.first, image_coordinates.second);
}

double* ImageData::GetMutableDataPointer(
    const int channel_index, const int pixel_index) const {

  CHECK_GE(channel_index, 0) << "Channel index must be at least 0.";
  CHECK_LT(channel_index, channels_.size()) << "Channel index out of bounds.";
  CHECK_GE(pixel_index, 0) << "Pixel index must be at least 0.";
  CHECK_LT(pixel_index, GetNumPixels()) << "Pixel index was out of bounds.";

  // TODO: verify that this is the correct approach of getting the data array.
  // static_cast doesn't work here because the data is apparently uchar*.
  double* pixel_data = (double*)(channels_[channel_index].data);  // NOLINT
  return pixel_data + pixel_index;
}

cv::Mat ImageData::GetVisualizationImage() const {
  cv::Mat visualization_image;
  if (channels_.empty()) {
    LOG(WARNING) << "This image is empty. Returning empty visualization image.";
    return visualization_image;
  }

  const int num_channels = channels_.size();
  if (num_channels < 3) {
    // For a monochrome image (or if it has two channels for some reason), just
    // return the first (and likely only) channel.
    visualization_image = channels_[0].clone();
    visualization_image.convertTo(visualization_image, CV_8UC1, 255);
  } else {
    // For 3 or more channels, return an RGB image of the first, middle, and
    // last channel. The middle channel is just the average index.
    std::vector<cv::Mat> bgr_channels = {
      channels_[0], channels_[num_channels / 2], channels_[num_channels - 1]
    };
    cv::merge(bgr_channels, visualization_image);
    visualization_image.convertTo(visualization_image, CV_8UC3, 255);
  }

  return visualization_image;
}

std::pair<int, int> ImageData::GetPixelCoordinatesFromIndex(
    const int index) const {

  CHECK_GE(index, 0) << "Pixel index must be at least 0.";
  CHECK_LT(index, GetNumPixels()) << "Pixel index was out of bounds.";

  const int row = index / image_size_.width;
  const int col = index % image_size_.width;
  return std::make_pair(row, col);
}

}  // namespace super_resolution
