#include "image/image_data.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// The actual implementation used by constructors ImageData(const cv::Mat&),
// ImageData(const cv::Mat&, const bool), and
// ImageData(const double*, const cv::Size&). The channels parameter should be
// the channels_ class variable.
//
// This method makes a copy of the given image data, so the original data or
// image is not modified when this ImageData is modified.
void InitializeFromImage(
    const cv::Mat& image,
    const bool normalize,
    cv::Size* image_size,
    std::vector<cv::Mat>* channels) {

  CHECK_NOTNULL(image_size);
  CHECK_NOTNULL(channels);

  *image_size = image.size();
  cv::split(image, *channels);  // makes a copy, even if just one channel
  for (int i = 0; i < channels->size(); ++i) {
    if (normalize) {
      (*channels)[i].convertTo(
          (*channels)[i],
          util::kOpenCvMatrixType,
          1.0 / 255.0);
    } else {
      (*channels)[i].convertTo((*channels)[i], util::kOpenCvMatrixType);
    }
  }
}

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
  // Make sure all pixels are within some valid range.
  double min_pixel_value, max_pixel_value;
  cv::minMaxLoc(image, &min_pixel_value, &max_pixel_value);
  CHECK_GE(min_pixel_value, 0)
      << "Invalid pixel range in given image: values cannot be negative.";
  CHECK_LE(max_pixel_value, 255)
      << "Invalid pixel range in given image: values cannot exceed 255.";

  const bool normalize = max_pixel_value > 1.0;
  InitializeFromImage(image, normalize, &image_size_, &channels_);
}

ImageData::ImageData(const cv::Mat& image, const bool normalize) {
  InitializeFromImage(image, normalize, &image_size_, &channels_);
}

ImageData::ImageData(
    const double* pixel_values, const cv::Size& size, const int num_channels) {

  CHECK_NOTNULL(pixel_values);
  CHECK_GE(num_channels, 1) << "The image must have at least one channel.";

  // Set image size and make sure the number of pixels is accurate.
  image_size_ = size;
  const int num_pixels = GetNumPixels();
  CHECK_GE(num_pixels, 1) << "Number of pixels must be positive.";

  // Add each channel to the ImageData.
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    const double* channel_pixels = &pixel_values[channel_index * num_pixels];
    const cv::Mat channel_image(
        size,
        util::kOpenCvMatrixType,
        const_cast<void*>(reinterpret_cast<const void*>(channel_pixels)));
    channels_.push_back(channel_image.clone());  // copy data
  }
}

void ImageData::AddChannel(const cv::Mat& channel_image) {
  // Make sure all pixels are within some valid range.
  double min_pixel_value, max_pixel_value;
  cv::minMaxLoc(channel_image, &min_pixel_value, &max_pixel_value);
  CHECK_GE(min_pixel_value, 0)
      << "Invalid pixel range in given image: values cannot be negative.";
  CHECK_LE(max_pixel_value, 255)
      << "Invalid pixel range in given image: values cannot exceed 255.";

  // Set or check size for consistency.
  if (channels_.empty()) {
    image_size_ = channel_image.size();
  } else {
    // Check size and type first if other channels already exist.
    CHECK(channel_image.size() == image_size_)
        << "Channel size did not match the expected size: "
        << channels_[0].size() << " size expected, "
        << channel_image.size() << " size given.";
  }

  cv::Mat converted_image = channel_image.clone();
  // Scale pixels between 0 and 1 if they are in the 0-255 range instead. Always
  // convert to the standard Matrix type in any case.
  if (max_pixel_value > 1.0) {
    converted_image.convertTo(
        converted_image, util::kOpenCvMatrixType, 1.0 / 255.0);
  } else {
    converted_image.convertTo(converted_image, util::kOpenCvMatrixType);
  }
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

void ImageData::UpsampleImage(const int scale_factor) {
  const cv::Size new_size = cv::Size(
      image_size_.width * scale_factor, image_size_.height * scale_factor);
  const int num_image_channels = channels_.size();
  for (int i = 0; i < num_image_channels; ++i) {
    const cv::Mat channel_image = channels_[i];
    // TODO: do the more efficient implementation here.
    cv::Mat resized_image = cv::Mat::zeros(new_size, channel_image.type());
    for (int row = 0; row < image_size_.height; ++row) {
      for (int col = 0; col < image_size_.width; ++col) {
        const int new_row = row * scale_factor;
        const int new_col = col * scale_factor;
        resized_image.at<double>(new_row, new_col) =
            channel_image.at<double>(row, col);
      }
    }
    channels_[i] = resized_image;
  }
  image_size_ = new_size;
}

int ImageData::GetNumPixels() const {
  return image_size_.width * image_size_.height;  // (0, 0) if image is empty.
}

cv::Mat ImageData::GetCroppedPatch(
    const int channel_index,
    const int pixel_index,
    const cv::Size& size) const {

  cv::Point anchor_coords = GetPixelCoordinatesFromIndex(pixel_index);
  const int radius_x = size.width / 2;
  const int radius_y = size.height / 2;
  int corner_x = anchor_coords.x - radius_x;
  // If the width is even, add 1 to offset the centering towards the left.
  if (size.width % 2 == 0) {
    corner_x += 1;
  }
  int corner_y = anchor_coords.y - radius_y;
  // If the height is even, add 1 to offset the centering towards the top.
  if (size.height % 2 == 0) {
    corner_y += 1;
  }

  const cv::Mat channel_image = GetChannelImage(channel_index);

  // Determine the width and height of the cropped patch since we can't crop
  // normally in cases where the patch extends the boundaries of the image.
  int crop_patch_width = size.width;  // Default width if it's within bounds.
  if (corner_x < 0) {
    // Corner x is negative, so take that much away from the width.
    crop_patch_width = size.width + corner_x;
  } else if (corner_x + size.width >= image_size_.width) {
    // Corner x is too close to the image width, only take what we can get.
    crop_patch_width = image_size_.width - corner_x;
  }
  // Same for the height.
  int crop_patch_height = size.height;
  if (corner_y < 0) {
    crop_patch_height = size.height + corner_y;
  } else if (corner_y + size.height >= image_size_.height) {
    crop_patch_height = image_size_.height - corner_y;
  }

  // Crop what we can from the image.
  const cv::Rect image_crop_region(
      std::max(0, corner_x),
      std::max(0, corner_y),
      crop_patch_width,
      crop_patch_height);
  const cv::Mat sub_patch = channel_image(image_crop_region);

  // If the sub-patch is the full patch, just return that.
  if (sub_patch.size() == size) {
    return sub_patch;
  }
  // Otherwise, pad zeros around the sub-patch before returning it.

  // Set up the patch region, anchored top-left at either at 0 or offset to
  // the right and/or down.
  const int patch_x = (corner_x < 0) ? -corner_x : 0;
  const int patch_y = (corner_y < 0) ? -corner_y : 0;
  const cv::Rect patch_region(
      patch_x, patch_y, image_crop_region.width, image_crop_region.height);

  // Copy the image sub-patch to the full patch in the appropriate region.
  // Everything not filled in will be padded with zeros.
  cv::Mat patch = cv::Mat::zeros(size, util::kOpenCvMatrixType);
  sub_patch.copyTo(patch(patch_region));
  return patch;
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

  const cv::Point image_coordinates =
      GetPixelCoordinatesFromIndex(pixel_index);  // Checks pixel index range.
  return channels_[channel_index].at<double>(
      image_coordinates.y, image_coordinates.x);
}

double* ImageData::GetMutableDataPointer(const int channel_index) const {
  CHECK_GE(channel_index, 0) << "Channel index must be at least 0.";
  CHECK_LT(channel_index, channels_.size()) << "Channel index out of bounds.";

  // TODO: verify that this is the correct approach of getting the data array.
  // static_cast doesn't work here because the data is apparently uchar*.
  return (double*)(channels_[channel_index].data);  // NOLINT
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

cv::Point ImageData::GetPixelCoordinatesFromIndex(const int index) const {
  CHECK_GE(index, 0) << "Pixel index must be at least 0.";
  CHECK_LT(index, GetNumPixels()) << "Pixel index was out of bounds.";

  const int x = index % image_size_.width;  // col
  const int y = index / image_size_.width;  // row
  return cv::Point(x, y);
}

}  // namespace super_resolution
