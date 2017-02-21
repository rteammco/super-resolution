#include "optimization/objective_irls_regularization_term.h"

#include <utility>
#include <vector>

#include "glog/logging.h"

namespace super_resolution {

double ObjectiveIrlsRegularizationTerm::Compute(
    const double* estimated_image_data, double* gradient) const {

  CHECK_NOTNULL(estimated_image_data);

  // Don't compute anything if the regularization parameter is 0.
  if (regularization_parameter_ <= 0.0) {
    return 0.0;
  }

  double residual_sum = 0.0;

  // Precompute the constant terms in the gradients at each pixel. This is
  // the regularization parameter (lambda) and the IRLS weights.
  // TODO: if gradient == nullptr, no need to compute the constants.
  const int num_pixels = image_size_.width * image_size_.height;
  const int num_data_points = num_pixels * num_channels_;
  std::vector<double> gradient_constants;
  gradient_constants.reserve(num_data_points);
  for (int i = 0; i < num_data_points; ++i) {
    gradient_constants.push_back(
        regularization_parameter_ * irls_weights_.at(i));
  }

  // Compute the residuals and squared residual sum.
  // TODO: just ApplyToImage if gradient == nullptr.
  const std::pair<std::vector<double>, std::vector<double>>&
  values_and_partials = regularizer_->ApplyToImageWithDifferentiation(
          estimated_image_data, gradient_constants);

  // The values are the regularizer values at each pixel and the partials are
  // the sum of partial derivatives at each pixel.
  const std::vector<double>& values = values_and_partials.first;
  const std::vector<double>& partials = values_and_partials.second;

  for (int i = 0; i < num_data_points; ++i) {
    const double residual = values[i];
    const double weight = irls_weights_.at(i);

    residual_sum += regularization_parameter_ * weight * residual * residual;
    // TODO: this still computes the gradient, just use ApplyToImage if
    // gradient is nullptr.
    if (gradient != nullptr) {
      gradient[i] += partials[i];
    }
  }

  return residual_sum;
}

}  // namespace super_resolution
