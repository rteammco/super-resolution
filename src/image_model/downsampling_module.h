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
  explicit DownsamplingModule(const int scale);

  virtual void ApplyToImage(ImageData* image_data, const int index) const;

  // The radius depends on the scale. For example, a 2x downsampling requires
  // at least a 2 x 2 grid to produce one pixel; hence, a radius of at least 1
  // (a 3x3 grid). A 3x downsampling operator would require a 3x3 grid, hence
  // the radius would be 1 also. A 4x downsampling would need a 4x4 grid,
  // resulting in a radius of 2, and same for 5x downsampling.
  virtual int GetPixelPatchRadius() const {
    return scale_ / 2;
  }

  // TODO: implement.
  virtual cv::Mat ApplyToPatch(
    const cv::Mat& patch,
    const int image_index,
    const int channel_index,
    const int pixel_index) const;

  virtual cv::Mat GetOperatorMatrix(
      const cv::Size& image_size, const int index) const;

 private:
  const int scale_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_
