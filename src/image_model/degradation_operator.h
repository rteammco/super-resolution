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
  // Virtual destructor for derived classes.
  virtual ~DegradationOperator() = default;

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

  // Returns the radius of the patch required to compute the degradation for
  // single pixel p. For example, a blur degradation with a 5 x 5 kernel needs
  // to be applied to a patch containing at least 2 pixels to the left, right,
  // top, and bottom of p. The radius would thus be 2.
  //
  // When using ApplyToPixel(), all other operators applied before this
  // operator must be applied to enough pixels so that the degradation can be
  // performed on a large enough spatial image patch.
  //
  // NOTE: the radius is the number of pixels surrounding the pixel to be
  // degraded, p. For example, radius 2 would imply a 5 x 5 patch, 2 on all
  // sides of p; radius 0 means that the operator does not require any spatial
  // information beyond the value of p itself.
  //
  // TODO: For efficiency, it may be useful to consider x, y radii separately.
  // TODO: Eventually, it may be even better to consider a spectral radius as
  //       well, i.e. (x, y, s).
  virtual int GetPixelPatchRadius() const = 0;

  // Applies the degradation to the given patch, which should be a sub-region
  // of the whole image. The patch must be at least big enough for the spatial
  // requirements of this operator (see GetPixelPatchRadius()).
  //
  // The returned patch will be the degraded version of the given patch, but
  // smaller by the patch degradation radius. That is, any pixels that only act
  // as a spatial dependency will not be degraded.
  //
  // For example, applying a 3x3 blurring kernel on a 5x5 patch will result in
  // a blurred 3x3 patch, since the 1-pixel-wide (radius = 1) border of the
  // image cannot be blurred but is needed to perform blurring of the other
  // pixels. In other words, the border cases of the degradation are thrown
  // out.
  //
  // This method should verify that the patch is sufficiently large to perform
  // the degradation.
  //
  // TODO: this may be obsolete due to changes in the solver.
  virtual cv::Mat ApplyToPatch(
      const cv::Mat& patch,
      const int image_index,
      const int channel_index,
      const int pixel_index) const = 0;

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
