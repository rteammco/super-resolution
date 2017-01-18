// An abstract interface for implementing regularization terms.

#ifndef SRC_SOLVERS_REGULARIZER_H_
#define SRC_SOLVERS_REGULARIZER_H_

#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

class Regularizer {
 public:
  // Initialize the regularization term object with the image size, which is
  // needed to compute the number of pixels, and in most implementations to
  // know the dimensions of the image.
  explicit Regularizer(const cv::Size& image_size) : image_size_(image_size) {}

  // Virtual destructor for derived classes.
  virtual ~Regularizer() = default;

  // Returns a vector of residuals based on the regularization for each pixel
  // in the given array.
  virtual std::vector<double> ApplyToImage(
      const double* image_data) const = 0;

  // Same as ApplyToImage, but the second vector returned in the pair contains
  // the derivative with respect to the residual at that index in the first
  // vector.  Derivatives computed using an automatic differentiation library.
  virtual std::pair<std::vector<double>, std::vector<double>>
  ApplyToImageWithDifferentiation(const double* image_data) const = 0;

  // Returns a vector of derivatives of the regularization term with respect to
  // each parameter in image_data. The given partial_const_terms vector
  // contains precomputed partial derivative constants from the full
  // regularization term in the objective function, and these contants will be
  // multiplied to each partial derivative computed for image_data.
  //
  // For each pixel i, computes the sum of partial derivatives dj/di for all
  // pixels j. The final derivative for each pixel i is defined as:
  //   d/di = sum_j(c[j] * dj/di)
  // where c[j] is the jth term in partial_const_terms.
  //
  // c[j] should include constants (e.g. 2 from differentiation a squared
  // term), the regularization parameter, etc.
  virtual std::vector<double> GetDerivatives(
      const double* image_data,
      const std::vector<double> partial_const_terms) const = 0;

 protected:
  // The size of the image to be regularized.
  const cv::Size image_size_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_REGULARIZER_H_
