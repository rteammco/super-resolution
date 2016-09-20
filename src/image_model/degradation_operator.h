// The DegradationOperator is a generic operator that somehow degrades an
// image, which can include warping/motion, blurring, or some form of noise. A
// chain of these operators defines a forward image model.
//
// Implement the degradation operators as needed by the specific application
// and form of the data.

#ifndef SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_
#define SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_

#include "opencv2/core/core.hpp"

namespace super_resolution {

class DegradationOperator {
 public:
  virtual void ApplyToImage(cv::Mat* image) const = 0;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_
