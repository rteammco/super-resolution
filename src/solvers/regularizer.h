// An abstract interface for implementing regularization terms.

#ifndef SRC_SOLVERS_REGULARIZER_H_
#define SRC_SOLVERS_REGULARIZER_H_

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

 protected:
  // The size of the image to be regularized.
  const cv::Size image_size_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_REGULARIZER_H_
