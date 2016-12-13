// A standard downsampling kernel that reduces the size of the given image.
// This module uses area interpolation, which does not use any linear or cubic
// interpolation methods. Instead it drops information from the high-resolution
// image to better simulate loss of data through low-resolution sensors. The
// downsampling scale is assumed to be the same in both the x and y directions.

#ifndef SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_
#define SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_

#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class DownsamplingModule : public DegradationOperator {
 public:
  // The given scale parameter is the scale by which the resized image will
  // be modified. The scale should be greater than or equal to 1.
  explicit DownsamplingModule(const int scale, const cv::Size& image_size);

  virtual void ApplyToImage(ImageData* image_data, const int index) const;

  virtual void ApplyTransposeToImage(
      ImageData* image_data, const int index) const;

  // The radius depends on the scale. For example, a 2x downsampling requires
  // at least 2 pixels - the center pixel and one other in any direction to
  // account for different subsampling shifts. Hence, a radius of at least 1 is
  // required. A 3x downsampling operator would require 2 pixels on either side
  // of the center pixel, so the radius would be 2.
  virtual int GetPixelPatchRadius() const {
    return scale_  - 1;
  }

  // TODO: implementation only works for scale = 2 and 3x3 patches for now.
  virtual cv::Mat ApplyToPatch(
    const cv::Mat& patch,
    const int image_index,
    const int channel_index,
    const int pixel_index) const;

  virtual cv::Mat GetOperatorMatrix(
      const cv::Size& image_size, const int index) const;

 private:
  // The downsampling scale.
  const int scale_;

  // The size of the image being downsampled. This is required for ApplyToPatch
  // method to compute the pixel's row and col from its index.
  const cv::Size image_size_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_
