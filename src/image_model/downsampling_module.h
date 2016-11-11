// A standard downsampling kernel that reduces the size of the given image.
// This module uses area interpolation, which does not use any linear or cubic
// interpolation methods. Instead it drops information from the high-resolution
// image to better simulate loss of data through low-resolution sensors. The
// downsampling scale is assumed to be the same in both the x and y directions.

#ifndef SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_
#define SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_

#include "image_model/degradation_operator.h"

namespace super_resolution {

class DownsamplingModule : public DegradationOperator {
 public:
  // The given scale parameter is the scale by which the resized image will
  // be modified. The scale should be greater than or equal to 1.
  explicit DownsamplingModule(const double scale);

  virtual void ApplyToImage(ImageData* image_data, const int index) const;

  virtual cv::Mat GetOperatorMatrix(
      const int num_pixels, const int index) const;

 private:
  const double scale_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_
