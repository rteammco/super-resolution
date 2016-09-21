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
  // Apply this degradation operator to the given image. The index is passed in
  // for cases where the degradation is dependent on the specific frame (e.g.
  // in the case of motion).
  virtual void ApplyToImage(cv::Mat* image, const int index) const = 0;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_
