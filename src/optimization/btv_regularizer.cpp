#include "optimization/btv_regularizer.h"

#include <utility>
#include <vector>

#include "glog/logging.h"

namespace super_resolution {

std::vector<double> BilateralTotalVariationRegularizer::ApplyToImage(
    const double* image_data) const {

  CHECK_NOTNULL(image_data);

  // TODO: implement.
  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> residuals(num_pixels * num_channels_);
  return residuals;
}

std::pair<std::vector<double>, std::vector<double>>
BilateralTotalVariationRegularizer::ApplyToImageWithDifferentiation(
    const double* image_data,
    const std::vector<double>& gradient_constants,
    const GradientComputationMethod& differentiation_method) const {

  CHECK_NOTNULL(image_data);

  // TODO: implement.
  const int num_pixels = image_size_.width * image_size_.height;
  const int num_parameters = num_pixels * num_channels_;
  std::vector<double> residual_values(num_parameters);
  std::vector<double> gradient(num_parameters);
  return std::make_pair(residual_values, gradient);
}

}  // namespace super_resolution
