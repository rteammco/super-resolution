// The bilateral total variation regularizer is a cheap-to-compute
// edge-preserving method for approximating the image gradient (i.e. standard
// total variation).

#ifndef SRC_OPTIMIZATION_BTV_REGULARIZER_H_
#define SRC_OPTIMIZATION_BTV_REGULARIZER_H_

#include <utility>
#include <vector>

#include "optimization/regularizer.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class BilateralTotalVariationRegularizer : public Regularizer {
 public:
  BilateralTotalVariationRegularizer(
      const cv::Size& image_size,
      const int num_channels,
      const int scale_range,
      const double spatial_decay);

  virtual std::vector<double> ApplyToImage(const double* image_data) const;

  virtual std::pair<std::vector<double>, std::vector<double>>
  ApplyToImageWithDifferentiation(
      const double* image_data,
      const std::vector<double>& gradient_constants,
      const GradientComputationMethod& differentiation_method =
          AUTOMATIC_DIFFERENTIATION) const;

 private:
  // The scale range controls the size of the patch that is checked for pixel
  // intensity variation.
  const int scale_range_;

  // The spatial decay parameter (0 < spatial_decay_ < 1) controls the weight
  // of each compared pixel. Pixels further from from the source pixel will
  // receive lower weights.
  //
  // Smaller spatial_decay_ values mean more decay as the pixels get further,
  // and larger values will make the decay minimal.
  const double spatial_decay_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_BTV_REGULARIZER_H_
