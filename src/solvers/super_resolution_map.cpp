#include "solvers/super_resolution_map.h"

#include <iostream>

#include "image/image_data.h"

namespace super_resolution {

ImageData MapSolver::Solve() const {
  // TODO: get an initial estimate.
  ImageData estimated_image;

  /* TODO: this is just "example" code in Ceres, if that is the solver to use.
   * This needs to be figured out and actually implemented.

  // TODO: how many channels?
  const int num_channels = 0;

  ceres::Problem problem;
  // TODO: currently solves independently for each channel.
  for (int channel = 0; channel < num_channels; ++channel) {
    for (const ImageData& observation : low_res_images_) {
      ceres::CostFunction* cost_function = MapCostFunction::Create(
          observation.GetPixelDataForChannel(channel), image_model_, channel);
      problem.AddResidualBlock(
          cost_function,
          loss_function,
          estimated_image.GetPixelDataForChannel(channel));
    }
  }

  // TODO: handle regularization.

  // Set the solver options. TODO: figure out what these should be.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  */

  return estimated_image;
}

}  // namespace super_resolution
