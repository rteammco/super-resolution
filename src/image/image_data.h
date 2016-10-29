// A generic image container for both regular and hyperspectral images. This
// container splits the image into independent channels (bands) of the image,
// each stored as an OpenCV Mat. This allows processing of hyperspectral images
// as well as RGB or monochrome images without modifying the code.

#ifndef SRC_IMAGE_IMAGE_DATA_H_
#define SRC_IMAGE_IMAGE_DATA_H_

#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

class ImageData {
 public:

  // Default constructor to make an empty image.
  ImageData();

  // Copy constructor clones the channel OpenCV Mats because they are
  // effectively smart pointers and are not copied by default.
  ImageData(const ImageData& other);

  // Pass in an OpenCV Mat to create an ImageData object out of that. If the
  // given image has multiple channels, they will all be added independently.
  explicit ImageData(const cv::Mat& image);

  // Appends a channel (band) to the image. Each new channel will be added as
  // the last index. Channel images should be single-band OpenCV images. The
  // added channel must have the same dimensions as the rest of the image.
  void AddChannel(const cv::Mat& channel_image);

  // Replaces a channel image with the given image. In most cases it is
  // possible to modify the cv::Mat directly (e.g. to blur it), but if the
  // image is resized a new matrix must be added to replace it. Error if the
  // given index is out of bounds.
  void ReplaceChannel(const cv::Mat& channel_image, const int index);

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

  // Returns the channel image (OpenCV Mat) at the given index. Error if index
  // is out of bounds. Use GetNumChannels() to get a valid range. Note that the
  // number of channels may be 0 for an empty image.
  cv::Mat GetChannelImage(const int index) const;

  // Returns the pixel value at the given channel and pixel indices. This will
  // be just a single intensity value for that specific pixel. The given
  // channel and pixel indices must be valid.
  double GetPixelValue(const int channel_index, const int pixel_index) const;

  // Returns a mutable data pointer (for the solver to adjust) at the given
  // channel and pixel indices.
  double* GetMutableDataPointer(
      const int channel_index, const int pixel_index) const;

  // Returns an OpenCV Mat image which is a naively-constructed monochrome or
  // RGB image combined from the channels in this image for visualization
  // purposes. An empty OpenCV Mat will be returned (and a warning will be
  // logged) if this image is empty.
  //
  // If the data is already monochrome or RGB, this will just return the image
  // in its original cv::Mat form. This can be used for storing or displaying
  // the data in native image form.
  cv::Mat GetVisualizationImage() const;

  // Returns the OpenCV type for the image (e.g. CV_64FC1). All channels stored
  // in this ImageData object have the same image type.
  //
  // The type is an implementation detail and is defined in image_data.cpp.
  // However, the pixel values are values between values of 0 and 1.
  int GetOpenCvImageType() const;

 private:
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
