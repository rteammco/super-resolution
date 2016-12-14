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

// Data struct that gets passed in as the pointer to the objective function and
// callback for processing the residuals and IRLS weights.
struct SolverMetaData {
  SolverMetaData(
      IrlsCostProcessor* irls_cost_processor,
      const int num_low_res_images,
      const int num_channels,
      const int num_pixels)
  : irls_cost_processor(irls_cost_processor),
    num_low_res_images(num_low_res_images),
    num_channels(num_channels),
    num_pixels(num_pixels) {}

  IrlsCostProcessor* irls_cost_processor;
  const int num_low_res_images;
  const int num_channels;
  const int num_pixels;
};

// The objective function used by the solver to compute residuals.
// TODO: compute derivatives manually since numerical diff takes way too long.
void ObjectiveFunction(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradients,  // NOLINT
    void* solver_meta_data_ptr) {

  const SolverMetaData* solver_meta_data =
      reinterpret_cast<const SolverMetaData*>(solver_meta_data_ptr);
  const IrlsCostProcessor* irls_cost_processor =
      solver_meta_data->irls_cost_processor;

  residual_sum = 0;

  // Initialize the gradients to 0.
  const int num_pixels = solver_meta_data->num_pixels;
  for (int i = 0; i < num_pixels; ++i) {
    gradients[i] = 0;
  }

  // Compute the data fidelity term residuals and the gradients.
  const int num_images = solver_meta_data->num_low_res_images;
  for (int image_index = 0; image_index < num_images; ++image_index) {
    // TODO: only channel 0 is currently supported. For channel 1+ offset data
    // pointer by num_pixels * channel_index.
    const int channel_index = 0;
    // TODO: IrlsCostProcessor should just return the number, no need to get a
    // vector and loop through it again.
    const std::vector<double> data_fidelity_residuals =
        irls_cost_processor->ComputeDataTermResiduals(
            image_index, channel_index, estimated_data.getcontent());
    for (const double residual : data_fidelity_residuals) {
      residual_sum += (residual * residual);
    }
    const std::vector<double> residual_derivatives =
        irls_cost_processor->ComputeDataTermDerivatives(
            image_index, data_fidelity_residuals.data());
    for (int i = 0; i < num_pixels; ++i) {
      gradients[i] += residual_derivatives[i];
    }
  }

  // Add the regularization term residuals.
  const std::vector<double> regularization_residuals =
      irls_cost_processor->ComputeRegularizationResiduals(
          estimated_data.getcontent());
  for (const double residual : regularization_residuals) {
    residual_sum += (residual * residual);
  }
  // TODO: manually compute derivatives here too:
  // irls_cost_processor->ComputeRegularizationDerivatives(...);
}

// The callback function, called after every solver iteration, which updates
// the IRLS weights.
void SolverIterationCallback(
    const alglib::real_1d_array& estimated_data,
    double residual_sum,
    void* solver_meta_data_ptr) {

  const SolverMetaData* solver_meta_data =
      reinterpret_cast<const SolverMetaData*>(solver_meta_data_ptr);
  IrlsCostProcessor* irls_cost_processor =
      solver_meta_data->irls_cost_processor;
  irls_cost_processor->UpdateIrlsWeights(estimated_data.getcontent());
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
  const alglib::ae_int_t max_num_iterations = 50;  // 0 = infinite.

  // TODO: multiple channel support?
  alglib::real_1d_array solver_data;
  solver_data.setcontent(
      num_pixels, initial_estimate.GetMutableDataPointer(0));

  alglib::mincgstate solver_state;
  alglib::mincgreport solver_report;
  alglib::mincgcreate(solver_data, solver_state);
  alglib::mincgsetcond(solver_state, epsg, epsf, epsx, max_num_iterations);
  alglib::mincgsetxrep(solver_state, true);

  // Solve and get results report.
  const SolverMetaData solver_meta_data(
      &irls_cost_processor, num_images, num_channels, num_pixels);
  alglib::mincgoptimize(
      solver_state,
      ObjectiveFunction,
      SolverIterationCallback,
      const_cast<void*>(reinterpret_cast<const void*>(&solver_meta_data)));
  alglib::mincgresults(solver_state, solver_data, solver_report);

  const ImageData estimated_image(
      solver_data.getcontent(), initial_estimate.GetImageSize());
  return estimated_image;
}

}  // namespace super_resolution
