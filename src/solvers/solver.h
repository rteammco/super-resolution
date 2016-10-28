// Defines an abstract (pure virtual) solver object that should be implemented
// with one or more specific super-resolution optimization methods (e.g. MAP or
// POCS).

#ifndef SRC_SOLVERS_SOLVER_H_
#define SRC_SOLVERS_SOLVER_H_

#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"

namespace super_resolution {

class Solver {
 public:
  Solver(
      const ImageModel& image_model,
      const std::vector<ImageData>& low_res_images)
    : image_model_(image_model), low_res_images_(low_res_images) {}

  // Solves the super-resolution optimization and returns the super-resolved
  // image. This must be implemented by the specific solver module.
  virtual ImageData Solve() const = 0;

 protected:
  const ImageModel& image_model_;
  const std::vector<ImageData>& low_res_images_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_SOLVER_H_
