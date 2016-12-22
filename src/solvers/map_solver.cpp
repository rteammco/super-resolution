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

// Computes the data fidelity term residuals and the gradient from the given
// estimated_data using the given IrlsCostProcessor. Pass in nullptr to the
// gradient array to avoid computing those (i.e. if doing numerical
// differentiation).
void ComputeDataTermResiduals(
    const int num_images,
    const double* estimated_data,
    const IrlsCostProcessor* irls_cost_processor,
    double* residual_sum,
    double* gradient = nullptr) {

  CHECK_NOTNULL(residual_sum);

  for (int image_index = 0; image_index < num_images; ++image_index) {
    // TODO: only channel 0 is currently supported. For channel 1+ offset data
    // pointer by num_pixels * channel_index.
    const int channel_index = 0;
    // TODO: IrlsCostProcessor should just return the number, no need to get a
    // vector and loop through it again.
    const std::vector<double> residuals =
        irls_cost_processor->ComputeDataTermResiduals(
            image_index, channel_index, estimated_data);
    for (const double residual : residuals) {
      *residual_sum += (residual * residual);
    }
    if (gradient != nullptr) {
      const int num_pixels = residuals.size();
      const std::vector<double> derivatives =
          irls_cost_processor->ComputeDataTermDerivatives(
              image_index, residuals.data());
      for (int i = 0; i < num_pixels; ++i) {
        gradient[i] += derivatives[i];
      }
    }
  }
}

// Add the regularization term residuals and derivatives.
void ComputeRegularizationResiduals(
    const double* estimated_data,
    const IrlsCostProcessor* irls_cost_processor,
    double* residual_sum,
    double* gradient = nullptr) {

  CHECK_NOTNULL(residual_sum);

  const std::vector<double> residuals =
      irls_cost_processor->ComputeRegularizationResiduals(estimated_data);
  for (const double residual : residuals) {
    *residual_sum += (residual * residual);
  }
  if (gradient != nullptr) {
    const int num_pixels = residuals.size();
    const std::vector<double> derivatives =
        irls_cost_processor->ComputeRegularizationDerivatives(estimated_data);
    for (int i = 0; i < num_pixels; ++i) {
      gradient[i] += derivatives[i];
    }
  }
}

// The objective function used by the solver to compute residuals. This version
// uses analyitical differentiation, meaning that the gradient is computed
// manually.
void ObjectiveFunctionAnalyticalDifferentiation(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    alglib::real_1d_array& gradient,  // NOLINT
    void* solver_meta_data_ptr) {

  const SolverMetaData* solver_meta_data =
      reinterpret_cast<const SolverMetaData*>(solver_meta_data_ptr);
  const IrlsCostProcessor* irls_cost_processor =
      solver_meta_data->irls_cost_processor;

  // Initialize the residuals and gradient values to 0.
  residual_sum = 0;
  for (int i = 0; i < solver_meta_data->num_pixels; ++i) {
    gradient[i] = 0;
  }

  // Compute the data fidelity term residuals and the gradient.
  ComputeDataTermResiduals(
      solver_meta_data->num_low_res_images,
      estimated_data.getcontent(),
      irls_cost_processor,
      &residual_sum,
      gradient.getcontent());

  // Compute the regularization residuals and the gradient.
  ComputeRegularizationResiduals(
    estimated_data.getcontent(),
    irls_cost_processor,
    &residual_sum,
    gradient.getcontent());
}

// A numerical differentiation version of the cost function, mostly for
// debugging purposes.
void ObjectiveFunctionNumericalDifferentiation(
    const alglib::real_1d_array& estimated_data,
    double& residual_sum,  // NOLINT
    void* solver_meta_data_ptr) {

  const SolverMetaData* solver_meta_data =
      reinterpret_cast<const SolverMetaData*>(solver_meta_data_ptr);
  const IrlsCostProcessor* irls_cost_processor =
      solver_meta_data->irls_cost_processor;

  residual_sum = 0;

  ComputeDataTermResiduals(
      solver_meta_data->num_low_res_images,
      estimated_data.getcontent(),
      irls_cost_processor,
      &residual_sum);

  ComputeRegularizationResiduals(
    estimated_data.getcontent(),
    irls_cost_processor,
    &residual_sum);
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
  const SolverMetaData solver_meta_data(
      &irls_cost_processor, num_images, num_channels, num_pixels);
  if (solver_options_.use_numerical_differentiation) {
    alglib::mincgoptimize(
        solver_state,
        ObjectiveFunctionNumericalDifferentiation,
        SolverIterationCallback,
        const_cast<void*>(reinterpret_cast<const void*>(&solver_meta_data)));
  } else {
    alglib::mincgoptimize(
        solver_state,
        ObjectiveFunctionAnalyticalDifferentiation,
        SolverIterationCallback,
        const_cast<void*>(reinterpret_cast<const void*>(&solver_meta_data)));
  }
  alglib::mincgresults(solver_state, solver_data, solver_report);

  const ImageData estimated_image(
      solver_data.getcontent(), initial_estimate.GetImageSize());
  return estimated_image;
}

}  // namespace super_resolution
