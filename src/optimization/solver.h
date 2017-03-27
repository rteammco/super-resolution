// Defines an abstract solver object that should be implemented with one or
// more specific super-resolution optimization methods (e.g. MAP or POCS).

#ifndef SRC_OPTIMIZATION_SOLVER_H_
#define SRC_OPTIMIZATION_SOLVER_H_

#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"

namespace super_resolution {

class Solver {
 public:
  explicit Solver(const ImageModel& image_model, const bool verbose = true)
      : image_model_(image_model), is_verbose_(verbose) {}

  // Solves the super-resolution optimization and returns the super-resolved
  // image. The given initial estimate is used as a starting point for
  // iterative methods.
  virtual ImageData Solve(const ImageData& initial_estimate) = 0;

  // Sets the output flag to false. If the derived class respects this
  // property, this will make the solver run silently.
  virtual void Stfu() {
    is_verbose_ = false;
  }

  // Returns true if the solver should print or log progress output. Use Stfu()
  // to silence the solver.
  virtual bool IsVerbose() const {
    return is_verbose_;
  }

 protected:
  const ImageModel& image_model_;

  // If set to false (true is the default), the solver should not print any
  // output (after or even during iterations) so that it runs silently. This
  // feature must be implemented by all derived classes to work.
  bool is_verbose_ = true;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_SOLVER_H_
