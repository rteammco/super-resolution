#include "solvers/map_solver.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "image/image_data.h"
#include "solvers/irls_cost_processor.h"
#include "solvers/regularizer.h"
#include "solvers/tv_regularizer.h"

#include "alglib/src/stdafx.h"
#include "alglib/src/optimization.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// The objective function used by the solver to compute residuals. This version
// uses analyitical differentiation, meaning that the gradient is computed
// manually.
void ObjectiveFunctionAnalyticalDifferentiation(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradient,  // NOLINT
    void* irls_cost_processor_ptr) {

  const IrlsCostProcessor* irls_cost_processor =
      reinterpret_cast<IrlsCostProcessor*>(irls_cost_processor_ptr);

  // Zero out the gradient (it isn't done automatically).
  const int num_pixels = irls_cost_processor->GetNumPixels();
  for (int i = 0; i < num_pixels; ++i) {
    gradient[i] = 0;
  }

  residual_sum = irls_cost_processor->ComputeObjectiveFunction(
      estimated_data.getcontent(), gradient.getcontent());
}

// A numerical differentiation version of the cost function, mostly for
// debugging purposes.
void ObjectiveFunctionNumericalDifferentiation(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    void* irls_cost_processor_ptr) {

  const IrlsCostProcessor* irls_cost_processor =
      reinterpret_cast<IrlsCostProcessor*>(irls_cost_processor_ptr);

  residual_sum = irls_cost_processor->ComputeObjectiveFunction(
      estimated_data.getcontent());
}

// The callback function, called after every solver iteration, which updates
// the IRLS weights.
void SolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* irls_cost_processor_ptr) {

  IrlsCostProcessor* irls_cost_processor =
      reinterpret_cast<IrlsCostProcessor*>(irls_cost_processor_ptr);
  irls_cost_processor->UpdateIrlsWeights(estimated_data.getcontent());
  LOG(INFO) << "Callback: residual sum = " << residual_sum;
}

ImageData MapSolver::Solve(const ImageData& initial_estimate) const {
  const int num_observations = low_res_images_.size();
  CHECK(num_observations > 0) << "Cannot solve with 0 low-res images.";

  const cv::Size hr_image_size = initial_estimate.GetImageSize();
  const int num_hr_pixels = initial_estimate.GetNumPixels();

  // Choose the regularizer according to the option selected.
  std::unique_ptr<Regularizer> regularizer;
  switch (solver_options_.regularization_method) {
    // TV is the default case.
    case TOTAL_VARIATION:
    default:
      regularizer = std::unique_ptr<Regularizer>(
          new TotalVariationRegularizer(hr_image_size));
  }

  IrlsCostProcessor irls_cost_processor(
      low_res_images_,
      image_model_,
      hr_image_size,
      std::move(regularizer),
      solver_options_.regularization_parameter);

  const int num_channels = low_res_images_[0].GetNumChannels();
  const int num_images = low_res_images_.size();
  const int num_pixels = initial_estimate.GetNumPixels();

  // Set up the optimization code with ALGLIB.
  // TODO: set these numbers correctly.
  const double epsg = 0.0000000001;
  const double epsf = 0.0;
  const double epsx = 0.0;
  const double numerical_diff_step_size = 1.0e-6;
  const alglib::ae_int_t max_num_iterations = 50;  // 0 = infinite.

  // TODO: multiple channel support?
  alglib::real_1d_array solver_data;
  solver_data.setcontent(
      num_pixels, initial_estimate.GetMutableDataPointer(0));

  alglib::mincgstate solver_state;
  alglib::mincgreport solver_report;
  if (solver_options_.use_numerical_differentiation) {
    alglib::mincgcreatef(solver_data, numerical_diff_step_size, solver_state);
  } else {
    alglib::mincgcreate(solver_data, solver_state);
  }
  alglib::mincgsetcond(solver_state, epsg, epsf, epsx, max_num_iterations);
  alglib::mincgsetxrep(solver_state, true);

  // Solve and get results report.
  if (solver_options_.use_numerical_differentiation) {
    alglib::mincgoptimize(
        solver_state,
        ObjectiveFunctionNumericalDifferentiation,
        SolverIterationCallback,
        reinterpret_cast<void*>(&irls_cost_processor));
  } else {
    alglib::mincgoptimize(
        solver_state,
        ObjectiveFunctionAnalyticalDifferentiation,
        SolverIterationCallback,
        reinterpret_cast<void*>(&irls_cost_processor));
  }
  alglib::mincgresults(solver_state, solver_data, solver_report);

  const ImageData estimated_image(
      solver_data.getcontent(), initial_estimate.GetImageSize());
  return estimated_image;
}

}  // namespace super_resolution
