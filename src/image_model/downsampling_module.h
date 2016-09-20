#ifndef SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_
#define SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_

#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class DownsamplingModule : public DegradationOperator {
 public:
  // The given scale parameter is the scale by which the resized image will
  // be modified.
  explicit DownsamplingModule(const double scale);

  virtual void ApplyToImage(cv::Mat* image) const;

 private:
  const double scale_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DOWNSAMPLING_MODULE_H_
