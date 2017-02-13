#include "optimization/irls_map_solver.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "optimization/alglib_irls_objective.h"

#include "alglib/src/optimization.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// Minimum residual value for computing IRLS weights, used to avoid division by
// zero.
constexpr double kMinResidualValue = 0.00001;

IrlsMapSolver::IrlsMapSolver(
    const MapSolverOptions& solver_options,
    const ImageModel& image_model,
    const std::vector<ImageData>& low_res_images,
    const bool print_solver_output)
    : MapSolver(
      solver_options, image_model, low_res_images, print_solver_output) {

  // Initialize IRLS weights to 1.
  irls_weights_.resize(GetNumPixels() * GetNumChannels());
  std::fill(irls_weights_.begin(), irls_weights_.end(), 1.0);
}

ImageData IrlsMapSolver::Solve(const ImageData& initial_estimate) {
  const int num_pixels = GetNumPixels();
  const int num_channels = GetNumChannels();
  CHECK_EQ(initial_estimate.GetNumPixels(), GetNumPixels());
  CHECK_EQ(initial_estimate.GetNumChannels(), GetNumChannels());

  // Reset all weights to 1.0.
  std::fill(irls_weights_.begin(), irls_weights_.end(), 1.0);

  // Set up the optimization code with ALGLIB.
  // Copy the initial estimate data to the solver's array.
  alglib::real_1d_array solver_data;
  const int num_data_points = GetNumDataPoints();
  solver_data.setlength(num_data_points);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    double* data_ptr = solver_data.getcontent() + (num_pixels * channel_index);
    const double* channel_ptr = initial_estimate.GetChannelData(channel_index);
    std::copy(channel_ptr, channel_ptr + num_pixels, data_ptr);
  }

  alglib::mincgstate solver_state;
  alglib::mincgreport solver_report;
  if (solver_options_.use_numerical_differentiation) {
    // Numerical differentiation setup.
    alglib::mincgcreatef(
        solver_data,
        solver_options_.numerical_differentiation_step,
        solver_state);
  } else {
    // Analytical differentiation setup.
    alglib::mincgcreate(solver_data, solver_state);
  }
  alglib::mincgsetcond(
      solver_state,
      solver_options_.gradient_norm_threshold,
      solver_options_.cost_decrease_threshold,
      solver_options_.parameter_variation_threshold,
      solver_options_.max_num_solver_iterations);
  alglib::mincgsetxrep(solver_state, true);

  // Solve and get results report.
  if (solver_options_.use_numerical_differentiation) {
    // Optimize with numerical differentiation.
    alglib::mincgoptimize(
        solver_state,
        AlglibObjectiveFunctionNumericalDiff,
        AlglibSolverIterationCallback,
        const_cast<void*>(reinterpret_cast<const void*>(this)));
  } else {
    // Optimize with analytical differentiation.
    alglib::mincgoptimize(
        solver_state,
        AlglibObjectiveFunction,
        AlglibSolverIterationCallback,
        const_cast<void*>(reinterpret_cast<const void*>(this)));
  }
  alglib::mincgresults(solver_state, solver_data, solver_report);

  const ImageData estimated_image(
      solver_data.getcontent(), GetImageSize(), num_channels);
  return estimated_image;
}

double IrlsMapSolver::ComputeDataTerm(
    const int image_index,
    const double* estimated_image_data,
    double* gradient) const {

  CHECK_NOTNULL(estimated_image_data);

  // Degrade (and re-upsample) the HR estimate with the image model.
  const int num_channels = GetNumChannels();
  const cv::Size image_size = GetImageSize();
  ImageData degraded_hr_image(estimated_image_data, image_size, num_channels);
  image_model_.ApplyToImage(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size, INTERPOLATE_NEAREST);

  // Compute the individual residuals by comparing pixel values. Sum them up
  // for the final residual sum.
  double residual_sum = 0;
  const int num_data_points = GetNumDataPoints();
  std::vector<double> residuals;
  residuals.reserve(num_data_points);
  const int num_pixels = GetNumPixels();
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    const double* degraded_hr_channel_data =
        degraded_hr_image.GetChannelData(channel_index);
    const double* observation_channel_data =
        observations_.at(image_index).GetChannelData(channel_index);
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      const double residual =
          degraded_hr_channel_data[pixel_index] -
          observation_channel_data[pixel_index];
      residuals.push_back(residual);
      residual_sum += (residual * residual);
    }
  }

  // If gradient is not null, apply transpose operations to the residual image.
  // This is used to compute the gradient.
  if (gradient != nullptr) {
    ImageData residual_image(residuals.data(), image_size, num_channels);
    const int scale = image_model_.GetDownsamplingScale();
    residual_image.ResizeImage(
        cv::Size(image_size.width / scale, image_size.height / scale),
        INTERPOLATE_NEAREST);
    image_model_.ApplyTransposeToImage(&residual_image, image_index);

    // Add to the gradient.
    for (int channel = 0; channel < num_channels; ++channel) {
      const int channel_index = channel * num_pixels;
      const double* residual_channel_data =
          residual_image.GetChannelData(channel);
      for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        const int index = channel_index + pixel_index;
        gradient[index] += 2 * residual_channel_data[pixel_index];
      }
    }
  }

  return residual_sum;
}

double IrlsMapSolver::ComputeRegularization(
    const double* estimated_image_data, double* gradient) const {

  CHECK_NOTNULL(estimated_image_data);

  const int num_data_points = GetNumDataPoints();

  // Apply each regularizer individually.
  double residual_sum = 0;
  for (const auto& regularizer_and_parameter : regularizers_) {
    const double regularization_parameter = regularizer_and_parameter.second;

    // Precompute the constant terms in the gradients at each pixel. This is
    // the regularization parameter (lambda) and the IRLS weights.
    std::vector<double> gradient_constants;
    gradient_constants.reserve(num_data_points);
    for (int i = 0; i < num_data_points; ++i) {
      gradient_constants.push_back(
          regularization_parameter * irls_weights_.at(i));
    }

    // Compute the residuals and squared residual sum.
    const std::pair<std::vector<double>, std::vector<double>>&
    values_and_partials =
        regularizer_and_parameter.first->ApplyToImageWithDifferentiation(
            estimated_image_data, gradient_constants);

    // The values are the regularizer values at each pixel and the partials are
    // the sum of partial derivatives at each pixel.
    const std::vector<double>& values = values_and_partials.first;
    const std::vector<double>& partials = values_and_partials.second;

    for (int i = 0; i < num_data_points; ++i) {
      const double residual = values[i];
      const double weight = irls_weights_.at(i);

      residual_sum += regularization_parameter * weight * residual * residual;
      // TODO: this still computes the gradient, just use ApplyToImage if
      // gradient is nullptr.
      if (gradient != nullptr) {
        gradient[i] += partials[i];
      }
    }
  }

  return residual_sum;
}

void IrlsMapSolver::UpdateIrlsWeights(const double* estimated_image_data) {
  CHECK_NOTNULL(estimated_image_data);

  // TODO: the regularizer is assumed to be L1 norm. Scale appropriately to L*
  // norm based on the regularizer's properties.
  // TODO: also, this assumes a single regularization term. Scale it up to more
  // (which means we need separate weights for each one).
  const int num_data_points = GetNumDataPoints();
  for (const auto& regularizer_and_parameter : regularizers_) {
    // TODO: maybe keep this from the computation phase to avoid recomputing?
    const std::vector<double>& regularization_residuals =
        regularizer_and_parameter.first->ApplyToImage(estimated_image_data);
    CHECK_EQ(regularization_residuals.size(), num_data_points)
        << "Number of residuals does not match number of weights.";
    for (int i = 0; i < num_data_points; ++i) {
      // TODO: this assumes L1 loss!
      // w = |r|^(p-2)
      irls_weights_[i] =
          1.0 / std::max(kMinResidualValue, regularization_residuals[i]);
    }
  }
}

}  // namespace super_resolution
