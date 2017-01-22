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
  using Regularizer::Regularizer;  // Inherit Regularizer constructor.

  // Implementation of total variation regularization.
  virtual std::vector<double> ApplyToImage(
      const double* image_data) const;

  virtual std::pair<std::vector<double>, std::vector<double>>
  ApplyToImageWithDifferentiation(
      const double* image_data,
      const std::vector<double>& gradient_constants,
      const GradientComputationMethod& differentiation_method =
          AUTOMATIC_DIFFERENTIATION) const;

 private:
  // TODO: fix.
  std::vector<double> GetGradient(
      const double* image_data,
      const std::vector<double>& gradient_constants) const;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_TV_REGULARIZER_H_
