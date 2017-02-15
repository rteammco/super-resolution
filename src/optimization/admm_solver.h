#ifndef SRC_OPTIMIZATION_ADMM_SOLVER_H_
#define SRC_OPTIMIZATION_ADMM_SOLVER_H_

#include "optimization/map_solver.h"

namespace super_resolution {

class AdmmSolver : public MapSolver {
 public:
  virtual ImageData Solve(const ImageData& initial_estimate);

};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_ADMM_SOLVER_H_
