#include "image/image_data.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// Returns true if the given ImageSpectralMode represents a 3-channel color
// image.
bool IsColorImage(const ImageSpectralMode& spectral_mode) {
  return (
      spectral_mode == SPECTRAL_MODE_COLOR_BGR ||
      spectral_mode == SPECTRAL_MODE_COLOR_YCRCB);
}

// Returns true if the given ImageSpectralMode represents a hyperspectral image
// with more than 3 channels.
bool IsHyperspectralImage(const ImageSpectralMode& spectral_mode) {
  return (
      spectral_mode == SPECTRAL_MODE_HYPERSPECTRAL ||
      spectral_mode == SPECTRAL_MODE_HYPERSPECTRAL_PCA);
}

// Returns the default spectral mode based on the number of channels (3-channels
// is BGR color, more is hyperspectral, otherwise none).
ImageSpectralMode GetDefaultSpectralMode(const int num_channels) {
  if (num_channels == 3) {
    return SPECTRAL_MODE_COLOR_BGR;
  } else if (num_channels > 3) {
    return SPECTRAL_MODE_HYPERSPECTRAL;
  } else {
    return SPECTRAL_MODE_NONE;
  }
}

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

// Resize each of the given image channels using additive interpolation (see
// the description of INTERPOLATE_ADDITIVE in image_data.h). If upsample is
// true, the scale will be used as an upsampling scale, otherwise it will be
// the downsampling scale. The new image size will be returned.
cv::Size ResizeAdditiveInterpolation(
    const cv::Size& new_size, std::vector<cv::Mat>* channels) {

  const int num_image_channels = channels->size();
  CHECK_GT(num_image_channels, 0)
      << "Cannot upsample an image with no channels.";

  const cv::Size original_size = channels->at(0).size();
  const bool upsample =
      original_size.width <= new_size.width &&
      original_size.height <= new_size.height;
  const bool downsample =
      original_size.width >= new_size.width &&
      original_size.height >= new_size.height;
  CHECK(upsample || downsample)
      << "Axis-independent up/downsampling is not supported.";

  // TODO: do the more efficient implementation? Also clean up, it's messy
  // repetitive code.
  if (upsample) {
    const int y_scale = new_size.height / original_size.height;
    const int x_scale = new_size.width / original_size.width;
    for (int i = 0; i < num_image_channels; ++i) {
      const cv::Mat channel_image = channels->at(i);
      cv::Mat resized_image = cv::Mat::zeros(new_size, channel_image.type());
      for (int row = 0; row < original_size.height; ++row) {
        for (int col = 0; col < original_size.width; ++col) {
          const int new_row = row * y_scale;
          const int new_col = col * x_scale;
          resized_image.at<double>(new_row, new_col) =
              channel_image.at<double>(row, col);
        }
      }
      (*channels)[i] = resized_image;
    }
    return new_size;
  } else {
    const int y_scale = original_size.height / new_size.height;
    const int x_scale = original_size.width / new_size.width;
    for (int i = 0; i < num_image_channels; ++i) {
      const cv::Mat channel_image = channels->at(i);
      cv::Mat resized_image = cv::Mat::zeros(new_size, channel_image.type());
      for (int row = 0; row < original_size.height; ++row) {
        for (int col = 0; col < original_size.width; ++col) {
          const int new_row = row / y_scale;
          const int new_col = col / x_scale;
          resized_image.at<double>(new_row, new_col) +=
              channel_image.at<double>(row, col);
        }
      }
      (*channels)[i] = resized_image;
    }
    return new_size;
  }
}

// Given two vectors, each with exactly 3 cv::Mat channels, interpolates the
// color components (channel 2 and 3) of the input into the output channels.
// The size of the channels will be made to match the size of the first (and
// unchanged) output channel (which is typically the luminance channel).
//
// This is used to interpolate the color components of the input into the
// generally higher-resolution structure-dominant output, such as when using a
// luminance-dominant color space such as YCrCb.
void InterpolateColor(
    const std::vector<cv::Mat>& input_channels,
    std::vector<cv::Mat>* output_channels) {

  CHECK_EQ(input_channels.size(), 3) << "Invalid number of input channels.";
  CHECK_EQ(output_channels->size(), 3) << "Invalid number of output channels.";

  const cv::Size target_size = output_channels->at(0).size();
  for (int i = 1; i < 3; ++i) {
    cv::Mat color_channel;
    // Only resize if the sizes are different.
    if (input_channels[i].size() != target_size) {
      cv::resize(
          input_channels[i],  // Source image.
          color_channel,      // Dest image.
          target_size,        // Desired image size.
          0,  // Set x, y scale to 0 to use the given Size instead.
          0,
          cv::INTER_LINEAR);
    } else {
      color_channel = input_channels[i].clone();
    }
    (*output_channels)[i] = color_channel;
  }
}

// ImageDataReport Print() method.
void ImageDataReport::Print() const {
  const int num_pixels = image_size.width * image_size.height * num_channels;
  const double percent_negative =
      (static_cast<double>(num_negative_pixels) /
      static_cast<double>(num_pixels)) * 100.0;
  const double percent_over_one =
      (static_cast<double>(num_over_one_pixels) /
      static_cast<double>(num_pixels)) * 100.0;
  std::cout << "Image Statistics: " << image_size.width
            << " x " << image_size.height << " x " << num_channels
            << " (" << num_pixels << " pixels)" << std::endl;
  std::cout << "  Num negative pixels: " << num_negative_pixels
            << " (" << percent_negative << "%)" << std::endl;
  std::cout << "  Num over one pixels: " << num_over_one_pixels
            << " (" << percent_over_one << "%)" << std::endl;
  std::cout << "  Channel with most negative pixels: "
            << channel_with_most_negative_pixels
            << " (" << max_num_negative_pixels_in_one_channel << ")"
            << std::endl;
  std::cout << "  Channel with most over one pixels: "
            << channel_with_most_over_one_pixels
            << " (" << max_num_over_one_pixels_in_one_channel << ")"
            << std::endl;
  std::cout << "  Minimum pixel value: " << smallest_pixel_value << std::endl;
  std::cout << "  Maximum pixel value: " << largest_pixel_value << std::endl;
}

// Default constructor.
ImageData::ImageData() {
  image_size_ = cv::Size(0, 0);
  spectral_mode_ = SPECTRAL_MODE_NONE;
}

// Copy constructor.
ImageData::ImageData(const ImageData& other)
    : spectral_mode_(other.spectral_mode_),
      luminance_channel_only_(other.luminance_channel_only_),
      image_size_(other.image_size_) {

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
      << "Invalid pixel range in given image: values cannot be negative. "
      << "Use ImageData(cv::Mat&, false) to avoid normalization, where any "
      << "image values are okay.";
  CHECK_LE(max_pixel_value, 255)
      << "Invalid pixel range in given image: values cannot exceed 255."
      << "Use ImageData(cv::Mat&, false) to avoid normalization, where any "
      << "image values are okay.";

  const bool normalize = max_pixel_value > 1.0;
  InitializeFromImage(image, normalize, &image_size_, &channels_);
  spectral_mode_ = GetDefaultSpectralMode(channels_.size());
}

ImageData::ImageData(const cv::Mat& image, const bool normalize) {
  InitializeFromImage(image, normalize, &image_size_, &channels_);
  spectral_mode_ = GetDefaultSpectralMode(channels_.size());
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
  spectral_mode_ = GetDefaultSpectralMode(channels_.size());
}

void ImageData::AddChannel(const cv::Mat& channel_image, const bool normalize) {
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
  double min_pixel_value, max_pixel_value;
  cv::minMaxLoc(channel_image, &min_pixel_value, &max_pixel_value);
  if (normalize && max_pixel_value > 1.0) {
    converted_image.convertTo(
        converted_image, util::kOpenCvMatrixType, 1.0 / 255.0);
  } else {
    converted_image.convertTo(converted_image, util::kOpenCvMatrixType);
  }
  channels_.push_back(converted_image);

  // Update color mode based on the number of channels now.
  spectral_mode_ = GetDefaultSpectralMode(channels_.size());
}

void ImageData::ResizeImage(
    const cv::Size& new_size,
    const ResizeInterpolationMethod interpolation_method) {

  // Undefined behavior if image is empty.
  CHECK(!channels_.empty()) << "Cannot resize an empty image.";
  CHECK_GT(new_size.width, 0) << "Images must have a positive width.";
  CHECK_GT(new_size.height, 0) << "Images must have a positive height.";

  int opencv_interpolation_method = 0;
  switch (interpolation_method) {
    case INTERPOLATE_ADDITIVE:
      // Custom implementation (not in OpenCV).
      image_size_ = ResizeAdditiveInterpolation(new_size, &channels_);
      return;
      break;
    case INTERPOLATE_LINEAR:
      opencv_interpolation_method = cv::INTER_LINEAR;
      break;
    case INTERPOLATE_CUBIC:
      opencv_interpolation_method = cv::INTER_CUBIC;
      break;
    case INTERPOLATE_NEAREST:
    default:
      opencv_interpolation_method = cv::INTER_NEAREST;
      break;
  }

  const int num_image_channels = GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat scaled_image;
    cv::resize(
        channels_[i],   // Source image.
        scaled_image,   // Dest image.
        new_size,       // Desired image size.
        0,              // Set x, y scale to 0 to use the given Size instead.
        0,
        opencv_interpolation_method);
    channels_[i] = scaled_image;
  }
  image_size_ = new_size;
}

void ImageData::ResizeImage(
    const double scale_factor,
    const ResizeInterpolationMethod interpolation_method) {

  // Undefined behavior if image is empty.
  CHECK(!channels_.empty()) << "Cannot resize an empty image.";
  CHECK_GT(scale_factor, 0) << "Scale factor must be larger than 0.";
  cv::Size new_size(
      static_cast<int>(image_size_.width * scale_factor),
      static_cast<int>(image_size_.height * scale_factor));
  ResizeImage(new_size, interpolation_method);
}

int ImageData::GetNumChannels() const {
  if (spectral_mode_ == SPECTRAL_MODE_COLOR_YCRCB && luminance_channel_only_) {
    return 1;
  }
  return channels_.size();
}

int ImageData::GetNumPixels() const {
  return image_size_.width * image_size_.height;  // (0, 0) if image is empty.
}

void ImageData::ChangeColorSpace(
    const ImageSpectralMode& new_color_mode, const bool luminance_only) {

  CHECK(IsColorImage(spectral_mode_))
      << "Cannot convert non-color (monochrome or hyperspectral) "
      << "images to a different color space.";

  CHECK(IsColorImage(new_color_mode))
      << "Invalid color space. new_color_mode must be SPECTRAL_MODE_COLOR_*.";

  // If it's already the same color mode, there's nothing to do.
  if (new_color_mode == spectral_mode_) {
    LOG(WARNING)
        << "This image is already set to the given color mode. "
        << "Image was not modified.";
    return;
  }

  // Set the OpenCV conversion mode value.
  int opencv_color_conversion_mode = 0;
  if (spectral_mode_ == SPECTRAL_MODE_COLOR_BGR &&
      new_color_mode == SPECTRAL_MODE_COLOR_YCRCB) {
    // BGR => YCrCb.
    opencv_color_conversion_mode = CV_BGR2YCrCb;
    luminance_channel_only_ = luminance_only;
  } else if (spectral_mode_ == SPECTRAL_MODE_COLOR_YCRCB &&
             new_color_mode == SPECTRAL_MODE_COLOR_BGR) {
    // YCrCb => BGR.
    opencv_color_conversion_mode = CV_YCrCb2BGR;
  } else {
    LOG(WARNING)
        << "Unsupported color mode: " << new_color_mode << ". "
        << "Image was not modified.";
    return;
  }

  // If going to BGR and luminance_channels_only_ is enabled, interpolate color
  // channels first to the appropriate size.
  if (new_color_mode == SPECTRAL_MODE_COLOR_BGR && luminance_channel_only_) {
    InterpolateColor(channels_, &channels_);
  }

  // Perform the conversion. Conversion is only supported in CV_32F mode, so we
  // need to convert to CV_32F and then back again.
  // Merge the 3 channels into a single cv::Mat image.
  cv::Mat converted_image;
  cv::merge(channels_, converted_image);
  // Convert to CV_32F format.
  const int original_type = converted_image.type();
  converted_image.convertTo(converted_image, CV_32F);
  // Convert to new color space.
  cv::cvtColor(converted_image, converted_image, opencv_color_conversion_mode);
  // Convert back to original format (double precision).
  converted_image.convertTo(converted_image, original_type);
  // Split the image back into individual ImageData channels.
  channels_.clear();
  cv::split(converted_image, channels_);

  spectral_mode_ = new_color_mode;
}

void ImageData::SetSpectralMode(const ImageSpectralMode& spectral_mode) {
  spectral_mode_ = spectral_mode;

  // Display some warnings if things don't line up.
  const int num_channels = GetNumChannels();
  const bool can_be_color =
      (num_channels == 3) || (num_channels == 1 && luminance_channel_only_);
  if (IsColorImage(spectral_mode) && !can_be_color) {
    std::string luminance_on_or_off;
    if (luminance_channel_only_) {
      luminance_on_or_off = "on";
    } else {
      luminance_on_or_off = "off";
    }
    LOG(WARNING)
        << "Spectral mode set to color but the image does not appear to be a "
        << "color image (" << num_channels << " channel(s), luminance-only = "
        << luminance_on_or_off << ").";
  }
  if (spectral_mode == SPECTRAL_MODE_HYPERSPECTRAL && num_channels <= 3) {
    LOG(WARNING)
        << "Spectral mode set to hyperspectral but number of spectra is too "
        << "low (" << num_channels << " spectral bands).";
  }
}

void ImageData::InterpolateColorFrom(const ImageData& color_image) {
  CHECK_EQ(GetNumChannels(), 1)  // If other 2 channels are hidden, ignore them.
      << "Color can only be interpolated for single-channel images.";
  CHECK_EQ(color_image.channels_.size(), 3)  // Consider hidden channels.
      << "The given image must have color information for interpolation.";

  channels_.resize(3);
  InterpolateColor(color_image.channels_, &channels_);
  spectral_mode_ = color_image.spectral_mode_;
  luminance_channel_only_ = false;
}

cv::Mat ImageData::GetChannelImage(const int index) const {
  CHECK_GE(index, 0) << "Channel index must be at least 0.";
  CHECK_LT(index, GetNumChannels()) << "Channel index out of bounds.";
  return channels_[index];
}

double ImageData::GetPixelValue(
    const int channel_index, const int pixel_index) const {

  const cv::Point image_coordinates =
      GetPixelCoordinatesFromIndex(pixel_index);  // Checks pixel index range.
  return GetPixelValue(channel_index, image_coordinates.y, image_coordinates.x);
}

double ImageData::GetPixelValue(
    const int channel_index, const int row, const int col) const {

  CHECK(0 <= channel_index && channel_index < GetNumChannels())
      << "Channel index is out of bounds.";
  CHECK(0 <= row && row < image_size_.height) << "Row index is out of bounds.";
  CHECK(0 <= col && col < image_size_.width) << "Col index is out of bounds.";

  return channels_[channel_index].at<double>(row, col);
}

const double* ImageData::GetChannelData(const int channel_index) const {
  return GetMutableChannelData(channel_index);
}

double* ImageData::GetMutableChannelData(const int channel_index) const {
  CHECK_GE(channel_index, 0) << "Channel index must be at least 0.";
  CHECK_LT(channel_index, GetNumChannels()) << "Channel index out of bounds.";

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

  const int num_channels = channels_.size();  // Consider all channels.
  if (num_channels < 3) {
    // For a monochrome image (or if it has two channels for some reason), just
    // return the first (and likely only) channel.
    visualization_image = channels_[0].clone();
    util::ThresholdImage(visualization_image, 0.0, 1.0);
    visualization_image.convertTo(visualization_image, CV_8UC1, 255);
  } else {
    // If the image is a 3-channel image and the color mode was not BGR,
    // convert the visualization to BGR first.
    if (!(IsHyperspectralImage(spectral_mode_) ||
          spectral_mode_ == SPECTRAL_MODE_COLOR_BGR)) {
      // TODO: this may be a bit hacky, but it works.
      ImageData converted_bgr_image = *this;  // Copy self.
      converted_bgr_image.ChangeColorSpace(SPECTRAL_MODE_COLOR_BGR);
      return converted_bgr_image.GetVisualizationImage();
    }
    // For 3 or more channels, return an RGB image of the first, middle, and
    // last channel. The middle channel is just the average index.
    std::vector<cv::Mat> bgr_channels = {
      channels_[0], channels_[num_channels / 2], channels_[num_channels - 1]
    };
    cv::merge(bgr_channels, visualization_image);
    util::ThresholdImage(visualization_image, 0.0, 1.0);
    visualization_image.convertTo(visualization_image, CV_8UC3, 255);
  }

  return visualization_image;
}

ImageDataReport ImageData::GetImageDataReport() const {
  ImageDataReport report;
  report.image_size = image_size_;
  report.num_channels = channels_.size();

  // Initialize these to the opposite extreme values so they can be adjusted.
  report.smallest_pixel_value = 1.0;
  report.largest_pixel_value = 0.0;

  for (int channel = 0; channel < channels_.size(); ++channel) {
    const cv::Mat& channel_image = channels_[channel];
    const int num_negative_pixels = cv::countNonZero(channel_image < 0.0);
    const int num_over_one_pixels = cv::countNonZero(channel_image > 1.0);
    if (num_negative_pixels > report.max_num_negative_pixels_in_one_channel) {
      report.channel_with_most_negative_pixels = channel;
      report.max_num_negative_pixels_in_one_channel = num_negative_pixels;
    }
    if (num_over_one_pixels > report.max_num_over_one_pixels_in_one_channel) {
      report.channel_with_most_over_one_pixels = channel;
      report.max_num_over_one_pixels_in_one_channel = num_over_one_pixels;
    }
    report.num_negative_pixels += num_negative_pixels;
    report.num_over_one_pixels += num_over_one_pixels;

    double min_channel_value, max_channel_value;
    cv::minMaxLoc(channel_image, &min_channel_value, &max_channel_value);
    report.smallest_pixel_value =
        std::min(min_channel_value, report.smallest_pixel_value);
    report.largest_pixel_value =
        std::max(max_channel_value, report.largest_pixel_value);
  }
  return report;
}

// private
cv::Point ImageData::GetPixelCoordinatesFromIndex(const int index) const {
  CHECK_GE(index, 0) << "Pixel index must be at least 0.";
  CHECK_LT(index, GetNumPixels()) << "Pixel index was out of bounds.";

  const int x = index % image_size_.width;  // col
  const int y = index / image_size_.width;  // row
  return cv::Point(x, y);
}

}  // namespace super_resolution
