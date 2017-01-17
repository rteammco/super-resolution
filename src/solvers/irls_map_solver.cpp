#include "solvers/irls_map_solver.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "solvers/ceres_objective.h"

#include "opencv2/core/core.hpp"

#include "ceres/ceres.h"

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
  // Reset all weights to 1.0.
  std::fill(irls_weights_.begin(), irls_weights_.end(), 1.0);

  const int num_channels = observations_[0].GetNumChannels();  // TODO: use!
  ImageData estimated_image = initial_estimate;

  ceres::GradientProblem problem(new MapDataCostFunction(this, 0));
  ceres::GradientProblemSolver::Options options;
  //options.update_state_every_iteration = true;
  options.callbacks.push_back(new MapIterationCallback());
  options.minimizer_progress_to_stdout = true;

  ceres::GradientProblemSolver::Summary summary;
  ceres::Solve(
      options,
      problem,
      estimated_image.GetMutableDataPointer(0),
      &summary);
  std::cout << summary.FullReport() << "\n";

//  // Set up the optimization code with Ceres.
//  ceres::Problem problem;
////  for (int image_index = 0; image_index < GetNumImages(); ++image_index) {
//    ceres::CostFunction* cost_function =
//        new MapDataCostFunction(this, 0);//image_index);
//    // TODO: channel 0 only, extend to multiple channels!
//    problem.AddResidualBlock(
//        cost_function,
//        NULL,
//        estimated_image.GetMutableDataPointer(0));
////  }
//
//  // TODO: handle regularization.
//
//  // Set the solver options. TODO: figure out what these should be.
//  ceres::Solver::Options options;
//  // options.linear_solver_type = ceres::DENSE_SCHUR;
//  // options.num_threads = 4;
//  // options.num_linear_solver_threads = 4;
//  // Always update parameters because we need to compute the new LR estimates.
//  options.update_state_every_iteration = true;
//  options.callbacks.push_back(new MapIterationCallback());
//  options.minimizer_progress_to_stdout = true;
//
//  // Solve.
//  ceres::Solver::Summary summary;
//  ceres::Solve(options, &problem, &summary);
//  std::cout << summary.FullReport() << std::endl;

  return estimated_image;
}

std::pair<double, std::vector<double>>
IrlsMapSolver::ComputeDataTermAnalyticalDiff(
    const int image_index,
    const int channel_index,
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  // Degrade (and re-upsample) the HR estimate with the image model.
  ImageData degraded_hr_image(estimated_image_data, image_size_);
  image_model_.ApplyToImage(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size_, cv::INTER_NEAREST);

  const int num_pixels = GetNumPixels();

  // Compute the individual residuals by comparing pixel values. Sum them up
  // for the final residual sum.
  double residual_sum = 0;
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    const double residual =
        degraded_hr_image.GetPixelValue(0, i) -
        observations_.at(image_index).GetPixelValue(channel_index, i);
    residuals.push_back(residual);
    residual_sum += (residual * residual);
  }

  // Apply transpose operations to the residual image. This is used to compute
  // the gradient.
  ImageData residual_image(residuals.data(), image_size_);
  const int scale = image_model_.GetDownsamplingScale();
  residual_image.ResizeImage(cv::Size(
      image_size_.width / scale,
      image_size_.height / scale));
  image_model_.ApplyTransposeToImage(&residual_image, image_index);

  // Build the gradient vector.
  std::vector<double> gradient(num_pixels);
  std::fill(gradient.begin(), gradient.end(), 0);
  for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
    // Only 1 channel (channel 0) in the residual image.
    const double partial_derivative =
        2 * residual_image.GetPixelValue(0, pixel_index);
    gradient[pixel_index] = partial_derivative;
  }

  return make_pair(residual_sum, gradient);
}

std::pair<double, std::vector<double>>
IrlsMapSolver::ComputeRegularizationAnalyticalDiff(
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  const int num_pixels = GetNumPixels();
  std::vector<double> gradient(num_pixels);
  std::fill(gradient.begin(), gradient.end(), 0);
  double residual_sum = 0;

  // Apply each regularizer individually.
  for (int i = 0; i < regularizers_.size(); ++i) {
    // Compute the residuals and squared residual sum.
    const double regularization_parameter = regularizers_[i].second;
    std::vector<double> residuals =
        regularizers_[i].first->ApplyToImage(estimated_image_data);

    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      const double weight = sqrt(irls_weights_.at(pixel_index));
      const double residual =
          regularization_parameter * weight * residuals[pixel_index];
      residuals[pixel_index] = residual;
      // TODO: ^^^
      //   Should we square this? Keep it unchanged? Maybe this is the issue?
      residual_sum += (residual * residual);
    }

    // Compute the gradient vector.
    std::vector<double> partial_const_terms;
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      // Each derivative is multiplied by
      //   2 * lambda * w^2 * reg_i
      // where 2 comes from the squared norm (L2) term,
      // lambda is the regularization parameter,
      // w^2 is the squared weight (since the weights are square-rooted in the
      //   residual computation, the raw weight is used here),
      // and reg_i is the value of the regularization term at pixel i.
      // These constants are multiplied with the partial derivatives at each
      // pixel w.r.t. all other pixels, which are computed specifically based on
      // the derivative of the regularizer function.
      partial_const_terms.push_back(
          2 *
          regularization_parameter *
          irls_weights_[pixel_index] *
          residuals[pixel_index]);
    }
    const std::vector<double> partial_derivatives =
        regularizers_[i].first->GetDerivatives(
            estimated_image_data, partial_const_terms);
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      gradient[pixel_index] += partial_derivatives[pixel_index];
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
