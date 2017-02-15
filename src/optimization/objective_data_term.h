// Defines the data term of the standard MAP objective function. This term
// computes ||Ax - y||_2^2 where A is the image model, x is the estimated data,
// and y is an observation. For multiple observations, the term is computed as
// the sum of costs over all observations k, ||A_kx - y_k||_2^2.

#ifndef SRC_OPTIMIZATION_OBJECTIVE_DATA_TERM_H_
#define SRC_OPTIMIZATION_OBJECTIVE_DATA_TERM_H_

#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "optimization/objective_function.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class ObjectiveDataTerm : public ObjectiveTerm {
 public:
  ObjectiveDataTerm(
      const ImageModel& image_model,
      const std::vector<ImageData>& observations,
      const int num_channels,
      const cv::Size& image_size)
    : image_model_(image_model),
      observations_(observations),
      num_channels_(num_channels),
      image_size_(image_size) {}

  virtual double Compute(
      const double* estimated_image_data, double* gradient) const;

 private:
  // The image model and observation information.
  const ImageModel& image_model_;
  const std::vector<ImageData>& observations_;
  const int num_channels_;
  const cv::Size& image_size_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_OBJECTIVE_DATA_TERM_H_
