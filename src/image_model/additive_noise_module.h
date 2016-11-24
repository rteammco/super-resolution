// A basic additive Gaussian noise module that adds random zero-mean noise to
// each pixel of the image.

#ifndef SRC_IMAGE_MODEL_ADDITIVE_NOISE_MODULE_H_
#define SRC_IMAGE_MODEL_ADDITIVE_NOISE_MODULE_H_

#include "image/image_data.h"
#include "image_model/degradation_operator.h"

namespace super_resolution {

class AdditiveNoiseModule : public DegradationOperator {
 public:
  // The additive noise is sampled from a zero-mean standard deviation with the
  // given sigma value (for pixel values between 0 to 255). Sigma must be
  // greater than 0.
  explicit AdditiveNoiseModule(const double sigma);

  virtual void ApplyToImage(ImageData* image_data, const int index) const;

  // Noise is applied independently per-pixel, so no need for any additional
  // spatial information.
  virtual int GetPixelPatchRadius() const {
    return 0;
  }

  // TODO: implement.
  virtual double ApplyToPixel(
    const ImageData& image_data,
    const int image_index,
    const int channel_index,
    const int pixel_index) const;

 private:
  const double sigma_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_ADDITIVE_NOISE_MODULE_H_
