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
  // Converts the given kernel to an operator matrix that can be applied to a
  // vectorized version of an image of the given size.
  //
  // This is a standard algorithm that can be used for any spatial filtering
  // kernel such as a Gaussian blur kernel.
  //
  // NOTE: Due to high computational and memory costs, this function limits the
  // given matrix size to at most 30 in each dimension (i.e. at most 30x30) and
  // similarly the kernel size to at most 10x10.
  static cv::Mat ConvertKernelToOperatorMatrix(
      const cv::Mat& kernel, const cv::Size& image_size);

  // Apply this degradation operator to the given image. The index is passed in
  // for cases where the degradation is dependent on the specific frame (e.g.
  // in the case of motion).
  virtual void ApplyToImage(ImageData* image_data, const int index) const = 0;

  // Returns a Matrix representation of this operator. The matrix is intended
  // to be applied onto a vectorized version of the image, assuming it is a
  // column vector of stacked rows. The image_size parameter is required for
  // some computations.
  //
  // This function by default returns a num_pixels by num_pixels identity
  // matrix (num_pixels is Size width * height), which will not change the
  // image vector.
  //
  // NOTE: This function can be very slow and is intended for testing with very
  // small data sets.
  virtual cv::Mat GetOperatorMatrix(
      const cv::Size& image_size, const int index) const;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_DEGRADATION_OPERATOR_H_
