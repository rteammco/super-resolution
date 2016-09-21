// A basic additive Gaussian noise module that adds random zero-mean noise to
// each pixel of the image.

#ifndef SRC_IMAGE_MODEL_ADDITIVE_NOISE_MODULE_H_
#define SRC_IMAGE_MODEL_ADDITIVE_NOISE_MODULE_H_

#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class AdditiveNoiseModule : public DegradationOperator {
 public:
  // The additive noise is sampled from a zero-mean standard deviation with the
  // given sigma value (in pixels).
  explicit AdditiveNoiseModule(const double sigma);

  virtual void ApplyToImage(cv::Mat* image, const int index) const;

 private:
  const double sigma_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_ADDITIVE_NOISE_MODULE_H_
