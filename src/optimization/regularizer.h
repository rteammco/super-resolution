// An abstract interface for implementing regularization terms.

#ifndef SRC_OPTIMIZATION_REGULARIZER_H_
#define SRC_OPTIMIZATION_REGULARIZER_H_

#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

// The type of differentiation method used in ApplyToImageWithDifferentiation.
enum GradientComputationMethod {
  AUTOMATIC,  // Uses automatic differentiation library.
  ANALYTICAL  // Direct analytical derivatives if implemented.
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
  // function). That is, this is the gradient with respect to each pixel.
  // Derivatives computed using either automatic differentiation or direct
  // analytical differentiation (if implemented) based on the given method
  // flag.
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
      const std::vector<double>& gradient_constants,
      const GradientComputationMethod& method = AUTOMATIC) const = 0;

 protected:
  // The size of the image to be regularized.
  const cv::Size image_size_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_REGULARIZER_H_
