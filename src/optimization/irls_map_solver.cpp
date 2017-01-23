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
  // TODO: set these numbers correctly.
  const double epsg = 0.0000000001;
  const double epsf = 0.0;
  const double epsx = 0.0;
  const double numerical_diff_step_size = 1.0e-6;
  const alglib::ae_int_t max_num_iterations = 50;  // 0 = infinite.

  // Copy the initial estimate data to the solver's array.
  alglib::real_1d_array solver_data;
  const int num_data_points = GetNumDataPoints();
  solver_data.setlength(num_data_points);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    double* data_ptr = solver_data.getcontent() + (num_pixels * channel_index);
    double* channel_ptr = initial_estimate.GetMutableDataPointer(channel_index);
    std::copy(channel_ptr, channel_ptr + num_pixels, data_ptr);
  }

  alglib::mincgstate solver_state;
  alglib::mincgreport solver_report;
  // TODO: enable numerical diff option?
  // if (solver_options_.use_numerical_differentiation) {
  // alglib::mincgcreatef(solver_data, numerical_diff_step_size, solver_state);
  // } else {
  alglib::mincgcreate(solver_data, solver_state);
  // }
  alglib::mincgsetcond(solver_state, epsg, epsf, epsx, max_num_iterations);
  alglib::mincgsetxrep(solver_state, true);

  // TODO: enable numerical diff option?
  // Solve and get results report.
  // if (solver_options_.use_numerical_differentiation) {
  //   alglib::mincgoptimize(
  //       solver_state,
  //       ObjectiveFunctionNumericalDifferentiation,
  //       SolverIterationCallback,
  //       const_cast<void*>(reinterpret_cast<const void*>(this)));
  // } else {
  alglib::mincgoptimize(
      solver_state,
      AlglibObjectiveFunction,
      AlglibSolverIterationCallback,
      const_cast<void*>(reinterpret_cast<const void*>(this)));
  // }
  alglib::mincgresults(solver_state, solver_data, solver_report);

  const ImageData estimated_image(
      solver_data.getcontent(), GetImageSize(), num_channels);
  return estimated_image;
}

std::pair<double, std::vector<double>> IrlsMapSolver::ComputeDataTerm(
    const int image_index,
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  // Degrade (and re-upsample) the HR estimate with the image model.
  const int num_channels = GetNumChannels();
  const cv::Size image_size = GetImageSize();
  ImageData degraded_hr_image(estimated_image_data, image_size, num_channels);
  image_model_.ApplyToImage(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size, cv::INTER_NEAREST);

  // Compute the individual residuals by comparing pixel values. Sum them up
  // for the final residual sum.
  double residual_sum = 0;
  const int num_data_points = GetNumDataPoints();
  std::vector<double> residuals(num_data_points);
  const int num_pixels = GetNumPixels();
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      const double residual =
          degraded_hr_image.GetPixelValue(channel_index, pixel_index) -
          observations_.at(image_index).GetPixelValue(
              channel_index, pixel_index);
      const int data_index = channel_index * num_pixels + pixel_index;
      residuals[data_index] = residual;
      residual_sum += (residual * residual);
    }
  }

  // Apply transpose operations to the residual image. This is used to compute
  // the gradient.
  ImageData residual_image(residuals.data(), image_size, num_channels);
  const int scale = image_model_.GetDownsamplingScale();
  residual_image.ResizeImage(cv::Size(
      image_size.width / scale,
      image_size.height / scale));
  image_model_.ApplyTransposeToImage(&residual_image, image_index);

  // Build the gradient vector.
  std::vector<double> gradient(num_data_points);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      const double gradient_at_pixel =
          2 * residual_image.GetPixelValue(channel_index, pixel_index);
      const int data_index = channel_index * num_pixels + pixel_index;
      gradient[data_index] = gradient_at_pixel;
    }
  }

  return make_pair(residual_sum, gradient);
}

std::pair<double, std::vector<double>> IrlsMapSolver::ComputeRegularization(
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  const int num_data_points = GetNumDataPoints();
  std::vector<double> gradient(num_data_points);

  // Apply each regularizer individually.
  double residual_sum = 0;
  for (const auto& regularizer_and_parameter : regularizers_) {
    const double regularization_parameter = regularizer_and_parameter.second;

    // Precompute the constant terms in the gradients at each pixel. This is
    // the regularization parameter (lambda) and the IRLS weights.
    std::vector<double> gradient_constants(num_data_points);
    for (int i = 0; i < num_data_points; ++i) {
      gradient_constants[i] = regularization_parameter * irls_weights_.at(i);
    }

    // Compute the residuals and squared residual sum.
    std::pair<std::vector<double>, std::vector<double>> values_and_partials =
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
      gradient[i] += partials[i];
    }
  }

  return make_pair(residual_sum, gradient);
}

void IrlsMapSolver::UpdateIrlsWeights(const double* estimated_image_data) {
  CHECK_NOTNULL(estimated_image_data);

  // TODO: the regularizer is assumed to be L1 norm. Scale appropriately to L*
  // norm based on the regularizer's properties.
  // TODO: also, this assumes a single regularization term. Scale it up to more
  // (which means we need separate weights for each one).
  const int num_data_points = GetNumDataPoints();
  for (const auto& regularizer_and_parameter : regularizers_) {
    std::vector<double> regularization_residuals =
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
