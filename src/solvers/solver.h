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
      const bool print_solver_output = true) :
    image_model_(image_model),
    print_solver_output_(print_solver_output) {}

  // Solves the super-resolution optimization and returns the super-resolved
  // image. The given initial estimate is used as a starting point for
  // iterative methods.
  virtual ImageData Solve(const ImageData& initial_estimate) const = 0;

  // Sets the output flag to false. If the derived class respects this
  // property, this will make the solver run silently.
  virtual void Stfu() {
    print_solver_output_ = false;
  }

 protected:
  const ImageModel& image_model_;

  // If set to false (true is the default), the solver should not print any
  // output (after or even during iterations) so that it runs silently. This
  // feature must be implemented by all derived classes to work.
  bool print_solver_output_ = true;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_SOLVER_H_
