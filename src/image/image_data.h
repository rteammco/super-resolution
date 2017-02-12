// A generic image container for both regular and hyperspectral images. This
// container splits the image into independent channels (bands) of the image,
// each stored as an OpenCV Mat. This allows processing of hyperspectral images
// as well as RGB or monochrome images without modifying the code.

#ifndef SRC_IMAGE_IMAGE_DATA_H_
#define SRC_IMAGE_IMAGE_DATA_H_

#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace super_resolution {

// The image color scheme. Only applicable to 3-channel color images. Single
// channel or hyperspectral images cannot be color-converted and should be
// identified as COLOR_MODE_NONE.
enum ImageColorMode {
  COLOR_MODE_NONE,
  COLOR_MODE_BGR,  // OpenCV uses BGR instead of RGB by default.
  COLOR_MODE_YCRCB,
  COLOR_MODE_YCRCB_LUMINANCE_ONLY  // Only consider the Yc channel for SR.
};

// Contains information and statistics about an image. This can be useful for
// evaluation, testing of new optimization methods, and debugging.
struct ImageDataReport {
  // Standardl image stats.
  cv::Size image_size;
  int num_channels = 0;

  // Number of invalid pixels.
  int num_negative_pixels = 0;  // Negative pixels are not valid values.
  int num_over_one_pixels = 0;  // Pixels that exceed maximum valid value (1.0).

  // Negative and over-one pixels per channel.
  int channel_with_most_negative_pixels = 0;
  int max_num_negative_pixels_in_one_channel = 0;
  int channel_with_most_over_one_pixels = 0;
  int max_num_over_one_pixels_in_one_channel = 0;

  double smallest_pixel_value = 0.0;
  double largest_pixel_value = 0.0;

  // Prints the report to standard output.
  void Print() const;
};

class ImageData {
 public:
  // Default constructor to make an empty image.
  ImageData();

  // Copy constructor clones the channel OpenCV Mats because they are
  // effectively smart pointers and are not copied by default.
  ImageData(const ImageData& other);

  // Pass in an OpenCV Mat to create an ImageData object out of that. If the
  // given image has multiple channels, they will all be added independently.
  // If the image is given in a non-normalized range (0-255 pixel values), it
  // will automatically be normalized to values between 0 and 1. All images
  // will be cloned and converted to a standard Mat type.
  explicit ImageData(const cv::Mat& image);

  // Same as the ImageData(const cv::Mat&) constructor, but does not
  // automatically normalize the values between 0 and 1. Instead, the user
  // explicitly indicates if the values should be normalized.
  //
  // Furthermore, this constructor allows pixel values in any range, even if
  // they are invalid (e.g. negative pixel values are allowed).
  //
  // TODO: for now it always assumes a range of 0-255 but that might not always
  // be the case.
  ImageData(const cv::Mat& image, const bool normalize);

  // Builds the ImageData directly from the given pixel value array, so the
  // user doesn't have to explicitly build a cv::Mat beforehand. The number of
  // pixels must match the given size width * height at each image channel.
  //
  // This constructor does not adjust the given pixel values in any way, so no
  // normalization happens.
  ImageData(
      const double* pixel_values,
      const cv::Size& size,
      const int num_channels = 1);

  // Appends a channel (band) to the image. Each new channel will be added as
  // the last index. Channel images should be single-band OpenCV images. The
  // added channel must have the same dimensions as the rest of the image.
  // Images given in a non-normalized range (0-255 pixel values) will
  // automatically be noramlized to values between 0 and 1.
  void AddChannel(const cv::Mat& channel_image);

  // Resizes this image to the given Size. The given Size must be valid (i.e.
  // positive values for width and height). All channels will be resized
  // equally. Any new channels added to this image must be the same size as the
  // rescaled image size. Empty images cannot be resized.
  //
  // NOTE: INTER_NEAREST is the "trivial" interpolation method which will just
  // select the nearest pixel to the downsampled pixel without doing any actual
  // interpolation (combining nearby pixel values). This method is preferable
  // for super-resolution downsampling to create the aliasing effect.
  void ResizeImage(
      const cv::Size& new_size,
      const int interpolation_method = cv::INTER_NEAREST);

  // Resizes this image by the given scale factor, in the same manner as
  // ResizeImage(size). The new dimensions will be (width * scale_factor,
  // height * scale_factor). The given scale factor must be larger than 0.
  void ResizeImage(
      const double scale_factor,
      const int interpolation_method = cv::INTER_NEAREST);

  // Resizes the image, does not do any interpolation. Instead, the pixels are
  // simply mapped to the high-res image, and the remaining pixels are padded
  // with zero. This is used to mimic the behavior of a transposed downsampling
  // matrix. For example, a 2x upsampling of the following image:
  //
  //   | 1 | 2 | 3 |
  //   | 4 | 5 | 6 |
  //
  // will look like this:
  //
  //   | 1 | 0 | 2 | 0 | 3 | 0 |
  //   | 0 | 0 | 0 | 0 | 0 | 0 |
  //   | 4 | 0 | 5 | 0 | 6 | 0 |
  //   | 0 | 0 | 0 | 0 | 0 | 0 |
  //
  // TODO: scale_factor may eventually be non-integer, but that is more
  // complicated.
  void UpsampleImage(const int scale_factor);

  // Returns the total number of channels (bands) in this image. Note that this
  // value may be 0.
  int GetNumChannels() const {
    return channels_.size();
  }

  // Returns the size of the image (width and height). If there are no channels
  // in this image (i.e. it is empty), the returned size will be (0, 0).
  cv::Size GetImageSize() const {
    return image_size_;
  }

  // Returns the number of pixels in the image at each channel. Each channel
  // has the same number of pixels. If the image is empty, 0 will be returned.
  int GetNumPixels() const;

  // Converts a 3-channel image into whichever OpenCV color space is specified.
  // This method can only be used on 3-channel color images. If the image does
  // not have exactly three channels, this will cause an error (check fail).
  void ChangeColorSpace(
      const ImageColorMode& new_color_mode, const bool luminance_dominant);

  // Returns the channel image (OpenCV Mat) at the given index. Error if index
  // is out of bounds. Use GetNumChannels() to get a valid range. Note that the
  // number of channels may be 0 for an empty image.
  cv::Mat GetChannelImage(const int index) const;

  // Returns the pixel value at the given channel and pixel indices. This will
  // be just a single intensity value for that specific pixel. The given
  // channel and pixel indices must be valid.
  double GetPixelValue(const int channel_index, const int pixel_index) const;

  // Same as GetPixelValue(channel_index, pixel_index) but allows the user to
  // specify the row, col coordinates instead of a pre-computed pixel index.
  double GetPixelValue(
      const int channel_index, const int row, const int col) const;

  // Returns a data pointer for the pixel values at the given channel index.
  // The size of the array will be the number of pixels in this image (use
  // GetNumPixels()).
  const double* GetChannelData(const int channel_index) const;

  // Same as GetChannelData(), but allows the image to be modified by changing
  // the values of the returned array.
  double* GetMutableChannelData(const int channel_index) const;

  // Returns an OpenCV Mat image which is a naively-constructed monochrome or
  // RGB image combined from the channels in this image for visualization
  // purposes. An empty OpenCV Mat will be returned (and a warning will be
  // logged) if this image is empty.
  //
  // If the data is already monochrome or RGB, this will just return the image
  // in its original cv::Mat form. This can be used for storing or displaying
  // the data in native image form.
  //
  // The visualization image will be re-scaled to standard pixel values of 0 to
  // 255.
  cv::Mat GetVisualizationImage() const;

  // Returns a ImageDataReport which contains information about the image that,
  // may be relevant to optimization. The data includes things like invalid
  // values (negative or larger than 1.0).
  ImageDataReport GetImageDataReport() const;

 private:
  // Returns a 2D pixel coordinate given the pixel index. This is used for
  // consistent indexing given a particular image size. The index range should
  // be (0 <= index < image_width * image_height) and will be verified.
  //
  // Pixels are accessed row-by-row: all pixels in the first row of the image,
  // followed by all pixels in the second row, etc. The returned coordinates
  // are (x [col], y [row]).
  cv::Point GetPixelCoordinatesFromIndex(const int index) const;

  // The color scheme of this image. If the image does not have exactly 3
  // channels, it will be specified COLOR_MODE_NONE. By default, it is assumed
  // that all images are represented in the BGR color space.
  //
  // Color mode is automatically updated in the constructors and AddChannel()
  // method based on the number of channels.
  ImageColorMode color_mode_;

  // The size of the image (width and height) is set when the first channel is
  // added. The size is guaranteed to be consistent between all channels in the
  // image. Empty images have a size of (0, 0).
  cv::Size image_size_;

  // The data is stored as OpenCV Mat images, one for each channel to support
  // an arbitrary number of channels.
  std::vector<cv::Mat> channels_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_IMAGE_DATA_H_
