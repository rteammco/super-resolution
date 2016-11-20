#include "solvers/regularizer.h"

#include "glog/logging.h"

namespace super_resolution {

Regularizer::Regularizer(const double lambda) : lambda_(lambda) {
  CHECK_GE(lambda_, 0) << "Regularization parameter must be non-negative.";
}

}  // namespace super_resolution
