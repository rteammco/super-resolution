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
  irls_weights_.resize(GetNumPixels());
  std::fill(irls_weights_.begin(), irls_weights_.end(), 1.0);
}

ImageData IrlsMapSolver::Solve(const ImageData& initial_estimate) {
  CHECK_EQ(initial_estimate.GetNumChannels(), num_channels_);

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
  const int num_pixels = GetNumPixels();
  const int num_data_points = num_pixels * num_channels_;
  solver_data.setlength(num_data_points);
  for (int channel_index = 0; channel_index < num_channels_; ++channel_index) {
    double* data_ptr = solver_data.getcontent() + (num_pixels * channel_index);
    std::copy(
        data_ptr,
        data_ptr + num_pixels,
        initial_estimate.GetMutableDataPointer(channel_index));
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
      solver_data.getcontent(), image_size_, num_channels_);
  return estimated_image;
}

std::pair<double, std::vector<double>> IrlsMapSolver::ComputeDataTerm(
    const int image_index,
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  // Degrade (and re-upsample) the HR estimate with the image model.
  ImageData degraded_hr_image(estimated_image_data, image_size_, num_channels_);
  image_model_.ApplyToImage(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size_, cv::INTER_NEAREST);

  const int num_pixels = GetNumPixels();

  // Compute the individual residuals by comparing pixel values. Sum them up
  // for the final residual sum.
  double residual_sum = 0;
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int channel_index = 0; channel_index < num_channels_; ++channel_index) {
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      const double residual =
          degraded_hr_image.GetPixelValue(channel_index, pixel_index) -
          observations_.at(image_index).GetPixelValue(
              channel_index, pixel_index);
      residuals.push_back(residual);
      residual_sum += (residual * residual);
    }
  }

  // Apply transpose operations to the residual image. This is used to compute
  // the gradient.
  ImageData residual_image(residuals.data(), image_size_, num_channels_);
  const int scale = image_model_.GetDownsamplingScale();
  residual_image.ResizeImage(cv::Size(
      image_size_.width / scale,
      image_size_.height / scale));
  image_model_.ApplyTransposeToImage(&residual_image, image_index);

  // Build the gradient vector.
  std::vector<double> gradient(num_pixels * num_channels_);
  std::fill(gradient.begin(), gradient.end(), 0);
  for (int channel_index = 0; channel_index < num_channels_; ++channel_index) {
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

  // TODO: multiple channels.
  const int num_pixels = GetNumPixels();
  std::vector<double> gradient(num_pixels);
  double residual_sum = 0;

  // Apply each regularizer individually.
  for (int i = 0; i < regularizers_.size(); ++i) {
    const double regularization_parameter = regularizers_[i].second;

    // Precompute the constant terms in the gradients at each pixel. This is
    // the regularization parameter (lambda) and the IRLS weights.
    std::vector<double> gradient_constants(num_pixels);
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      gradient_constants[pixel_index] =
          regularization_parameter * irls_weights_.at(pixel_index);
    }

    // Compute the residuals and squared residual sum.
    std::pair<std::vector<double>, std::vector<double>> values_and_partials =
        regularizers_[i].first->ApplyToImageWithDifferentiation(
            estimated_image_data, gradient_constants);

    // The values are the regularizer values at each pixel and the partials are
    // the sum of partial derivatives at each pixel.
    const std::vector<double>& values = values_and_partials.first;
    const std::vector<double>& partials = values_and_partials.second;

    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      const double residual = values[pixel_index];
      const double weight = irls_weights_.at(pixel_index);

      residual_sum += regularization_parameter * weight * residual * residual;

      gradient[pixel_index] += partials[pixel_index];
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
  const int num_pixels = GetNumPixels();
  for (int i = 0; i < regularizers_.size(); ++i) {
    std::vector<double> regularization_residuals =
        regularizers_[i].first->ApplyToImage(estimated_image_data);
    CHECK_EQ(regularization_residuals.size(), num_pixels)
        << "Number of residuals does not match number of weights.";
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      // TODO: this assumes L1 loss!
      // w = |r|^(p-2)
      irls_weights_[pixel_index] = 1.0 / std::max(
          kMinResidualValue, regularization_residuals[pixel_index]);
    }
  }
}

}  // namespace super_resolution
