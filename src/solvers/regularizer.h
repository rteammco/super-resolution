// An abstract interface for implementing regularization terms.

#ifndef SRC_SOLVERS_REGULARIZER_H_
#define SRC_SOLVERS_REGULARIZER_H_

#include <vector>

namespace super_resolution {

class Regularizer {
 public:
  // Initialize the regularization term object. "lambda" is the regularization
  // parameter, and it must be non-negative. If lambda = 0, this regularization
  // term effectively does nothing.
  explicit Regularizer(const double lambda);

  // Returns a vector of residuals
  virtual std::vector<double> ComputeResiduals(
      const double* image_data) const = 0;

 private:
  // The regularization parameter scales the residuals by this amount.
  const double lambda_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_REGULARIZER_H_
