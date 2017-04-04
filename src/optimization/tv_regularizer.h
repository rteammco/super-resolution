// The total variation regularization cost implementation. Total variation is
// effectively defined as the gradient value at each pixel. Its purpose is to
// add denoising by imposing smoothness in the estimated image (smaller changes
// between neighrboing pixels in the x and y directions).

#ifndef SRC_OPTIMIZATION_TV_REGULARIZER_H_
#define SRC_OPTIMIZATION_TV_REGULARIZER_H_

#include <utility>
#include <vector>

#include "optimization/regularizer.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class TotalVariationRegularizer : public Regularizer {
 public:
  // Constructor for using the 1-norm (taking the absolute value of the
  // gradient at each pixel). Using the 1-norm version is recommended.
  explicit TotalVariationRegularizer(const cv::Size& image_size)
      : Regularizer(image_size), use_3d_total_variation_(false) {}

  // Implementation of total variation regularization.
  virtual std::vector<double> ApplyToImage(
      const double* image_data, const int num_channels) const;

  virtual std::pair<std::vector<double>, std::vector<double>>
  ApplyToImageWithDifferentiation(
      const double* image_data,
      const std::vector<double>& gradient_constants,
      const int num_channels) const;

  // Turn using 3D total variation on or off. 3D TV may be preferable for
  // hyperspectral data and can be used experimentally for color images.
  void SetUse3dTotalVariation(const bool use_3d_total_variation) {
    use_3d_total_variation_ = use_3d_total_variation;
  }

 private:
  // If this is set to true, the regularizer will use 3D total variation (also
  // looking at the spectral direction instead of just the X, Y spatial
  // directions). This is a trivial extension of 2D total variation.
  bool use_3d_total_variation_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_TV_REGULARIZER_H_
