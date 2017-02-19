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

enum ResizeInterpolationMethod {
  // Standard interpolation modes:
  INTERPOLATE_LINEAR,  // Bilinear interpolation. Uses cv::INTER_LINEAR.
  INTERPOLATE_CUBIC,   // Bicubic interpolation. Uses cv::INTER_CUBIC.

  // Nearest neighbor (i.e. no interpolation). Uses cv::INTER_NEAREST.
  //
  // Upsampling: naively samples from the nearest neighbor in the LR image.
  //
  // | a | b |  2x =>  | a | a | b | b |
  // | c | d |         | a | a | b | b |
  //                   | c | c | d | d |
  //                   | c | c | d | d |
  //
  // Downsampling: chooses the top-left pixel in each patch of the HR image to
  // map to the LR grid. This method causes aliasing.
  //
  // | a | b | c | d |  2x =>  | a | c |
  // | e | f | g | h |         | i | k |
  // | i | j | k | l |
  // | m | n | o | p |
  INTERPOLATE_NEAREST,

  // When downsampling, additive interpolation sums all the pixels that overlap
  // the lower-resolution pixel. For upsampling, pads with zeros.
  //
  // Downsampling: each pixel in the LR image will be the sum of pixels in the
  // HR image patch that maps to it.
  //
  // | a | b | c | d |  2x =>  | a + b + e + f | c + d + g + h |
  // | e | f | g | h |         | i + j + m + n | k + l + o + p |
  // | i | j | k | l |
  // | m | n | o | p |
  //
  // Upsampling: just pads zeros around each element. The downsampling of an
  // upsampled image recovers the LR image exactly.
  //
  // | a | b |  2x =>  | a | 0 | b | 0 |
  // | c | d |         | 0 | 0 | 0 | 0 |
  //                   | c | 0 | d | 0 |
  //                   | 0 | 0 | 0 | 0 |
  INTERPOLATE_ADDITIVE
};

// The image spectral mode. For color images, this is used to determine the
// color space (default BGR). For hyperspectral images, this is used to
// determine if the image has been converted into a basis (e.g. using PCA).
enum ImageSpectralMode {
  SPECTRAL_MODE_NONE,               // Monochrome or undefined.

  // Hyperspectral images (more than 3 channels):
  SPECTRAL_MODE_HYPERSPECTRAL,      // Regular hyperspectral image, unchanged.
  SPECTRAL_MODE_HYPERSPECTRAL_PCA,  // PCA basis of hyperspectral bands.

  // Color modes (3-channel color images only):
  SPECTRAL_MODE_COLOR_BGR,          // OpenCV uses BGR, not RGB, by default.
  SPECTRAL_MODE_COLOR_YCRCB         // Luminance-dominant color.
};

// Contains information and statistics about an image. This can be useful for
// evaluation, testing of new optimization methods, and debugging.
struct ImageDataReport {
  // Standard image stats.
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
      const ResizeInterpolationMethod interpolation_method
          = INTERPOLATE_NEAREST);

  // Resizes this image by the given scale factor, in the same manner as
  // ResizeImage(size). The new dimensions will be (width * scale_factor,
  // height * scale_factor). The given scale factor must be larger than 0.
  void ResizeImage(
      const double scale_factor,
      const ResizeInterpolationMethod interpolation_method
          = INTERPOLATE_NEAREST);

  // Returns the total number of channels (bands) in this image. Note that this
  // value may be 0.
  //
  // If the color mode of this ImageData is set to a luminance-dominant color
  // space such as YCrCb and luminance_channel_only_ is enabled, the number of
  // channels will be reported as 1 instead of 3. The other two channels will
  // be hidden until the image is converted back to the BGR color space.
  int GetNumChannels() const;

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
  //
  // Set luminance_only = true to make this image only use the luminance
  // channel for super-resolution. This is only applicable to color spaces such
  // as YCrCb which have a luminance channel. If this is set, the image will be
  // treated as a single-channel image for the purposes of the solver.
  void ChangeColorSpace(
      const ImageSpectralMode& new_color_mode,
      const bool luminance_only = false);

  // This method will interpolate the color information from the given image
  // into this monochrome image. Typically, this image would be higher
  // resolution than the other given image so that structure is preserved and
  // color is interpolated from the lower-resolution image. Only applicable to
  // single-channel images, otherwise this will cause an error.
  //
  // This will inherit the color space from the given color image. For
  // luminance dominant color spaces, such as YCrCb, the single channel in this
  // ImageData will be treated as the luminance channel, and the other two
  // channels from the color_image will be interpolated. The user should ensure
  // the color spaces match.
  //
  // TODO: currently only works for YCrCb images.
  void InterpolateColorFrom(const ImageData& color_image);

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

  // The spectral mode of this image. SPECTRAL_MODE_COLOR_* is for 3-channel
  // color images. By default, it is assumed that all 3-channel images are
  // represented in the BGR color space. All images with more than 3 channels
  // will be assumed to be regular hyperspectral images.
  //
  // The spectral mode is automatically updated in the constructors and
  // AddChannel() method based on the number of channels.
  ImageSpectralMode spectral_mode_;

  // If the color mode is set to a color space that has a dominant luminance
  // channel, such as the YCrCb color space, then this flag indicates how the
  // image should be treated in super-resolution.
  //
  // If true, this ImageData will only expose the luminance channel to outside
  // methods (e.g. GetNumChannels() will return 1). If the image is then
  // converted back to the BGR color space, the color components (e.g. Cr and
  // Cb in the YCrCb color space) will be interpolated to the same size as the
  // luminance channel and the 3-channel BGR image will be reconstructed. This
  // is intended to save time and avoid color discontinuities when applying a
  // super-resolution algorithm.
  //
  // To enable this option, set the second argument in ChangeColorSpace() to
  // true (e.g. "ChangeColorSpace(COLOR_MODE_YCRCB, true)"). This will only
  // work if the supported color mode has a dominant luminance channel.
  bool luminance_channel_only_;

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
