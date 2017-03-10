// Defines the data term of the standard MAP objective function. This term
// computes ||Ax - y||_2^2 where A is the image model, x is the estimated data,
// and y is an observation. For multiple observations, the term is computed as
// the sum of costs over all observations k, ||A_kx - y_k||_2^2.

#ifndef SRC_OPTIMIZATION_OBJECTIVE_IRLS_REGULARIZATION_TERM_H_
#define SRC_OPTIMIZATION_OBJECTIVE_IRLS_REGULARIZATION_TERM_H_

#include <vector>

#include "optimization/objective_function.h"
#include "optimization/regularizer.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class ObjectiveIRLSRegularizationTerm : public ObjectiveTerm {
 public:
  ObjectiveIRLSRegularizationTerm(
      const std::shared_ptr<Regularizer> regularizer,
      const double regularization_parameter,
      const std::vector<double>& irls_weights,
      const int num_channels,
      const cv::Size& image_size)
    : regularizer_(regularizer),
      regularization_parameter_(regularization_parameter),
      irls_weights_(irls_weights),
      num_channels_(num_channels),
      image_size_(image_size) {}

  virtual double Compute(
      const double* estimated_image_data, double* gradient) const;

 private:
  const std::shared_ptr<Regularizer> regularizer_;
  const double regularization_parameter_;
  const std::vector<double>& irls_weights_;
  const int num_channels_;
  const cv::Size& image_size_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_OBJECTIVE_IRLS_REGULARIZATION_TERM_H_
