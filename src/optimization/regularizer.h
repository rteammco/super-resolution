// An abstract interface for implementing regularization terms.

#ifndef SRC_OPTIMIZATION_REGULARIZER_H_
#define SRC_OPTIMIZATION_REGULARIZER_H_

#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

// The type of differentiation method used in ApplyToImageWithDifferentiation.
enum GradientComputationMethod {
  AUTOMATIC_DIFFERENTIATION,  // Uses automatic differentiation library.
  ANALYTICAL_DIFFERENTIATION  // Direct analytical derivatives if implemented.
};

class Regularizer {
 public:
  // Initialize the regularization term object with the image size, which is
  // needed to compute the number of pixels, and in most implementations to
  // know the dimensions of the image.
  explicit Regularizer(const cv::Size& image_size) : image_size_(image_size) {}

  // Virtual destructor for derived classes.
  virtual ~Regularizer() = default;

  // Returns a vector of resulting values based on the regularization for each
  // pixel in the given image data array. This is NOT the final residual, but
  // contains the evaluation values at each pixel.
  virtual std::vector<double> ApplyToImage(
      const double* image_data) const = 0;

  // Same as ApplyToImage, but the second vector returned in the pair contains
  // the gradient (assuming a single residual value) in the objective
  // function). Derivatives computed using automatic differentiation.
  //
  // The given gradient_constants vector should contain for each pixel any
  // constant terms that will be multiplied with the gradient. This should
  // include the regularization parameter as well as any additional weights
  // used in schemes like reweighted least squares. If there are no constants
  // to multiply, the value should be set to 1.
  //
  // TODO: This currently works with 2-norm least squares gradients, but that
  // may change if the objective function uses a different norm.
  virtual std::pair<std::vector<double>, std::vector<double>>
  ApplyToImageWithDifferentiation(
      const double* image_data,
      const std::vector<double>& gradient_constants) const = 0;

  // TODO: Replace this method with the above method, .
  // Returns a vector of derivatives of the regularization term with respect to
  // each parameter in image_data. The given gravector
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
  virtual std::vector<double> GetGradient(
      const double* image_data,
      const std::vector<double>& gradient_constants) const = 0;

 protected:
  // The size of the image to be regularized.
  const cv::Size image_size_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_REGULARIZER_H_
