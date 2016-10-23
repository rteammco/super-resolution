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
  // Empty constructor initializes an empty image. Add channels to empty images
  // one at a time.
  ImageData() {}

  // Pass in an OpenCV Mat to create an ImageData object out of that.
  explicit ImageData(const cv::Mat& image);

  // Appends a channel (band) to the image. Each new channel will be added as
  // the last index. Channel images should be single-band OpenCV images. The
  // added channel must have the same dimensions as the rest of the image.
  void AddChannel(const cv::Mat& channel_image);

  // Returns the total number of channels (bands) in this image. Note that this
  // value may be 0.
  int GetNumChannels() const {
    return channels_.size();
  }

  // Returns the size of the image (width and height). If there are no channels
  // in this image (i.e. it is empty), the returned size will be (0, 0).
  cv::Size GetImageSize() const;

  // Returns the channel at the given index. Error if index is out of bounds.
  // Use GetNumChannels() to get a valid range. Note that the number of
  // channels may be 0 for an empty image.
  cv::Mat GetChannel(const int index) const;

  // Returns a visualization image. The visualization image is a
  // naively-constructed monochrome or RGB image combined from the channels in
  // this image for visualization purposes only. An empty OpenCV Mat will be
  // returned (and a warning will be logged) if this image is empty.
  cv::Mat GetVisualizationImage() const;

  // Returns the OpenCV type for the image (e.g. CV_16SC1). All channels have
  // the same image type. If this image is empty, -1 will be returned instead.
  int GetOpenCvType() const;

 private:
  std::vector<cv::Mat> channels_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_IMAGE_DATA_H_
