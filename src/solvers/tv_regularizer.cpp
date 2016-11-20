#include "solvers/tv_regularizer.h"

#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

TotalVariationRegularizer::TotalVariationRegularizer(
    const double lambda, const cv::Size& image_size)
  : Regularizer(lambda), image_size_(image_size) {}

std::vector<double> TotalVariationRegularizer::ComputeResiduals(
    const double* image_data) const {

  // TODO: implement
  std::vector<double> residuals;
  return residuals;
}

}  // namespace super_resolution
