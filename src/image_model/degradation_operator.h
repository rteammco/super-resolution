// The DegradationOperator is a generic operator that somehow degrades an
// image, which can include warping/motion, blurring, or some form of noise. A
// chain of these operators defines a forward image model.
//
// Implement the degradation operators as needed by the specific application
// and form of the data.

#ifndef SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_
#define SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class DegradationOperator {
 public:
  // Apply this degradation operator to the given image. The index is passed in
  // for cases where the degradation is dependent on the specific frame (e.g.
  // in the case of motion).
  virtual void ApplyToImage(ImageData* image_data, const int index) const = 0;

  // Returns a Matrix representation of this operator. The matrix is intended
  // to be applied onto a vectorized version of the image, assuming it is a
  // column vector of stacked rows. The num_pixels parameter indicates how many
  // pixels the image has (the size of its column vector).
  //
  // This function is implemented, and by default returns a num_pixels by
  // num_pixels identity matrix, which will not change the image vector.
  virtual cv::Mat GetOperatorMatrix(
      const int num_pixels, const int index) const;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_
