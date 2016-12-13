// A standard blurring kernel that applies a Gaussian blur, emulating a point
// spread function (PSF). The PSF is assumed to be the same in both the x and y
// directions.

#ifndef SRC_IMAGE_MODEL_PSF_BLUR_MODULE_H_
#define SRC_IMAGE_MODEL_PSF_BLUR_MODULE_H_

#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class PsfBlurModule : public DegradationOperator {
 public:
  // The given blur radius and sigma (in pixels) will define the Gaussian blur.
  // The blur radius must be at least 1 and sigma must be greater than 0.
  // The blur radius must be an odd number.
  PsfBlurModule(const int blur_radius, const double sigma);

  virtual void ApplyToImage(ImageData* image_data, const int index) const;

  virtual void ApplyTransposeToImage(
      ImageData* image_data, const int index) const;

  // Depends on the blur radius.
  virtual int GetPixelPatchRadius() const {
    return blur_radius_;
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
  const int blur_radius_;
  const double sigma_;

  // This kernel is created in the constructor and is used for the blurring
  // convolution and for getting the operator matrix.
  cv::Mat blur_kernel_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_PSF_BLUR_MODULE_H_
