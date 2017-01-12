#include "solvers/irls_map_solver.h"

#include <utility>
#include <vector>

#include "image/image_data.h"

#include "glog/logging.h"

namespace super_resolution {

ImageData IrlsMapSolver::Solve(const ImageData& initial_estimate) const {
  const int num_observations = low_res_images_.size();
  CHECK(num_observations > 0) << "Cannot solve with 0 low-res images.";

  // TODO: implement.
  return initial_estimate;
}

std::pair<double, std::vector<double>>
IrlsMapSolver::ComputeDataTermAnalyticalDiff(
    const int image_index,
    const int channel_index,
    const double* estimated_data) const {

  // TODO: implement.
  double residual_sum = 0;
  std::vector<double> gradient(16 /* TODO: # of pixels */);
  return make_pair(residual_sum, gradient);
}

}  // namespace super_resolution
