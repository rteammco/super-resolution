#include "solvers/tv_regularizer.h"

#include <vector>

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

TotalVariationRegularizer::TotalVariationRegularizer(
    const double lambda, const cv::Size& image_size)
  : Regularizer(lambda), image_size_(image_size) {}

std::vector<double> TotalVariationRegularizer::ComputeResiduals(
    const double* image_data) const {

  CHECK_NOTNULL(image_data);
  // TODO: implement
  // lambda_
  std::vector<double> residuals;
  residuals.resize(image_size_.width * image_size_.height);
  return residuals;
}

}  // namespace super_resolution
