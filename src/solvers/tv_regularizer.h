// The total variation regularization term.

#ifndef SRC_SOLVERS_TV_REGULARIZER_H_
#define SRC_SOLVERS_TV_REGULARIZER_H_

#include <vector>

#include "solvers/regularizer.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class TotalVariationRegularizer : public Regularizer {
 public:
  TotalVariationRegularizer(const double lambda, const cv::Size& image_size);

  virtual std::vector<double> ComputeResiduals(
      const double* image_data) const;

 private:
  const cv::Size image_size_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_TV_REGULARIZER_H_
