#include "image/image_data.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

ImageData::ImageData(const cv::Mat& image) {
  const int num_image_channels = image.channels();
  cv::split(image, channels_);
}

ImageData::ImageData(const ImageData& other) {
  for (const cv::Mat& channel_image : other.channels_) {
    channels_.push_back(channel_image.clone());
  }
}

void ImageData::AddChannel(const cv::Mat& channel_image) {
  // Check size and type first.
  if (!channels_.empty()) {
    CHECK(channel_image.size() == channels_[0].size())
        << "Channel size did not match the expected size: "
        << channels_[0].size() << " size expected, "
        << channel_image.size() << " size given.";
    CHECK(channel_image.type() == channels_[0].type())
        << "Channel type did not match the expected type: "
        << channels_[0].type() << " type expected, "
        << channel_image.type() << " type given.";
  } channels_.push_back(channel_image);
}

cv::Size ImageData::GetImageSize() const {
  // Return (0, 0) if this image is empty.
  if (channels_.empty()) {
    return cv::Size(0, 0);
  }
  // All channels must be the same size, so return the size of the first
  // channel.
  return channels_[0].size();
}

cv::Mat ImageData::GetChannel(const int index) const {
  CHECK_GE(index, 0) << "Minimum channel index is 0.";
  CHECK_LT(index, channels_.size())
      << "Index out of bounds: there are only "
      << channels_.size() << " image channels.";
  return channels_[index];
}

cv::Mat ImageData::GetVisualizationImage() const {
  // TODO: implement.
  cv::Mat visualization_image;
  if (channels_.empty()) {
    LOG(WARNING) << "This image is empty. Returning empty visualization image.";
  }
  return visualization_image;
}

int ImageData::GetOpenCvType() const {
  // Return -1 if this image is empty.
  if (channels_.empty()) {
    return -1;
  }
  return channels_[0].type();
}

}  // namespace super_resolution
