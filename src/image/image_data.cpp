#include "image/image_data.h"

#include <vector>

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// The type used for all OpenCV images stored in this object. Any images given
// in another format will be converted to this type.
constexpr int kOpenCvImageType = CV_64FC1;

// Default constructor.
ImageData::ImageData() {
  image_size_ = cv::Size(0, 0);
}

// Copy constructor.
ImageData::ImageData(const ImageData& other) : image_size_(other.image_size_) {
  // TODO: remove after verifying that this is indeed size 0 on copy.
  CHECK(channels_.empty());
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
    channels_[i].convertTo(channels_[i], kOpenCvImageType, 1.0 / 255.0);
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
    CHECK(channel_image.type() == channels_[0].type())
        << "Channel type did not match the expected type: "
        << channels_[0].type() << " type expected, "
        << channel_image.type() << " type given.";
  }

  // TODO: this only works if the given pixel values are 0 to 255.
  // TODO: make sure that this is making a copy of the image.
  cv::Mat converted_image;
  channel_image.convertTo(converted_image, kOpenCvImageType, 1.0 / 255.0);
  channels_.push_back(converted_image);
}

void ImageData::ResizeImage(
    const double scale_factor, const int interpolation_method) {

  CHECK_GT(scale_factor, 0) << "Scale factor must be larger than 0.";

  return;
  const int num_image_channels = channels_.size();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat scaled_image;
    cv::resize(
        channels_[i],     // Source image.
        scaled_image,     // Dest image.
        cv::Size(0, 0),   // Size is set to 0, so it will use the ratio.
        scale_factor,     // Scaling ratio in the x asix.
        scale_factor,     // Scaling ratio in the y axis.
        interpolation_method);
    channels_[i] = scaled_image;
  }

  // Update the size. If image was empty, the size was 0 and won't change.
  image_size_.width *= scale_factor;
  image_size_.height *= scale_factor;
}

int ImageData::GetNumPixels() const {
  return image_size_.width * image_size_.height;  // (0, 0) if image is empty.
}

cv::Mat ImageData::GetChannelImage(const int index) const {
  CHECK_GE(index, 0) << "Minimum channel index is 0.";
  CHECK_LT(index, channels_.size())
      << "Index out of bounds: there are only "
      << channels_.size() << " image channels.";
  return channels_[index];
}

double ImageData::GetPixelValue(
    const int channel_index, const int pixel_index) const {

  // TODO: implement and check index ranges.
  return 0.0;
}

double* ImageData::GetMutableDataPointer(
    const int channel_index, const int pixel_index) const {

  // TODO: implement and check index ranges.
  return nullptr;
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

int ImageData::GetOpenCvImageType() const {
  return kOpenCvImageType;
}

}  // namespace super_resolution
