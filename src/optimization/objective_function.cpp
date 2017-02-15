#include "optimization/objective_function.h"

namespace super_resolution {

double ObjectiveFunction::ComputeAllTerms(
    const double* estimated_image_data, double* gradient) const {

  // Reset gradient to 0 (if applicable).
  if (gradient != nullptr) {
    for (int i = 0; i < num_parameters_; ++i) {
      gradient[i] = 0.0;
    }
  }

  double residual_sum = 0.0;
  for (const std::shared_ptr<ObjectiveTerm> term : terms_) {
    residual_sum += term->Compute(estimated_image_data, gradient);
  }
  return residual_sum;
}

}  // namespace super_resolution
