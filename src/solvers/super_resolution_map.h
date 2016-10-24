#ifndef SRC_SOLVERS_SUPER_RESOLUTION_MAP_H_
#define SRC_SOLVERS_SUPER_RESOLUTION_MAP_H_

#include "image/image_data.h"
#include "solvers/solver.h"

namespace super_resolution {

class MapSolver : public Solver {
 public:
  // Solves the problem using the MAP (maximum a posteriori) formulation.
  virtual ImageData Solve() const;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_SUPER_RESOLUTION_MAP_H_
