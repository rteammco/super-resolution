// The bilateral total variation regularizer is a cheap-to-compute
// edge-preserving method for approximating the image gradient (i.e. standard
// total variation).

#ifndef SRC_OPTIMIZATION_BTV_REGULARIZER_H_
#define SRC_OPTIMIZATION_BTV_REGULARIZER_H_

#include <utility>
#include <vector>

#include "optimization/regularizer.h"

namespace super_resolution {

class BilateralTotalVariationRegularizer : public Regularizer {
 public:
  using Regularizer::Regularizer;

  virtual std::vector<double> ApplyToImage(const double* image_data) const;

  virtual std::pair<std::vector<double>, std::vector<double>>
  ApplyToImageWithDifferentiation(
      const double* image_data,
      const std::vector<double>& gradient_constants,
      const GradientComputationMethod& differentiation_method =
          AUTOMATIC_DIFFERENTIATION) const;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_BTV_REGULARIZER_H_
