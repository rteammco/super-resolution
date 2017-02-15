#include "optimization/irls_map_solver.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "optimization/alglib_objective.h"
#include "optimization/objective_data_term.h"
#include "optimization/objective_function.h"
#include "optimization/objective_irls_regularization_term.h"

#include "alglib/src/optimization.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

// Minimum residual value for computing IRLS weights, used to avoid division by
// zero.
constexpr double kMinResidualValue = 0.00001;

IrlsMapSolver::IrlsMapSolver(
    const IrlsMapSolverOptions& solver_options,
    const ImageModel& image_model,
    const std::vector<ImageData>& low_res_images,
    const bool print_solver_output)
    : MapSolver(image_model, low_res_images, print_solver_output),
      solver_options_(solver_options) {

  // Initialize IRLS weights to 1.
  irls_weights_.resize(GetNumDataPoints());
  std::fill(irls_weights_.begin(), irls_weights_.end(), 1.0);
}

ImageData IrlsMapSolver::Solve(const ImageData& initial_estimate) {
  const int num_pixels = GetNumPixels();
  const int num_channels = GetNumChannels();
  CHECK_EQ(initial_estimate.GetNumPixels(), GetNumPixels());
  CHECK_EQ(initial_estimate.GetNumChannels(), GetNumChannels());

  const int num_data_points = GetNumDataPoints();
  // TODO: here
  ObjectiveFunction objective_function_data_term_only(num_data_points);
  std::shared_ptr<ObjectiveTerm> data_term(new ObjectiveDataTerm(
      image_model_, observations_, num_channels, GetImageSize())); 
  objective_function_data_term_only.AddTerm(data_term);
// TODO: add regularizers.
//  std::shared_ptr<ObjectiveTerm> regularization_term(
//      new ObjectiveRegularizationTerm(regularizer));
  // TODO:
  // 1. Add regularizers, each loop iteration w/ new weights.
  // 2. Change AlglibObjectiveFunction to take the ObjectiveFunction object
  //    instead of this IrlsMapSolver.
  // 3. Change it all, and make sure it works.
  // 4. Clean up code.
  // 5. Implement ADMM.

  // Initialize the IRLS weights to 1.0.
  std::fill(irls_weights_.begin(), irls_weights_.end(), 1.0);

  // Set up the optimization code with ALGLIB.
  // Copy the initial estimate data to the solver's array.
  alglib::real_1d_array solver_data;
  solver_data.setlength(num_data_points);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    double* data_ptr = solver_data.getcontent() + (num_pixels * channel_index);
    const double* channel_ptr = initial_estimate.GetChannelData(channel_index);
    std::copy(channel_ptr, channel_ptr + num_pixels, data_ptr);
  }

  // Do the IRLS loop. After every iteration, update the IRLS weights and solve
  // again until the change in residual sum is sufficiently low.
  double previous_cost = std::numeric_limits<double>::infinity();
  double cost_difference = solver_options_.cost_decrease_threshold + 1.0;
  int num_iterations_ran = 0;
  while (std::abs(cost_difference) >=
         solver_options_.irls_cost_difference_threshold) {
    ObjectiveFunction objective_function = objective_function_data_term_only;
    for (const auto& regularizer_and_parameter : regularizers_) {
      std::shared_ptr<ObjectiveTerm> regularization_term(
          new ObjectiveIrlsRegularizationTerm(
              regularizer_and_parameter.first,
              regularizer_and_parameter.second,
              irls_weights_,
              num_channels,
              GetImageSize()));
      objective_function.AddTerm(regularization_term);
    }
    if (solver_options_.use_numerical_differentiation) {
      RunCGSolverNumericalDiff(
          solver_options_, objective_function, &solver_data);
    } else {
      RunCGSolverAnalyticalDiff(
          solver_options_, objective_function, &solver_data);
    }

    // Update the IRLS weights.
    // TODO: should this be computed off of the initial estimate? That seems to
    // get better results at the cost of A LOT of extra computational time.
    // TODO: the regularizer is assumed to be L1 norm. Scale appropriately to
    // L* norm based on the regularizer's properties.
    // TODO: also, this assumes a single regularization term. Scale it up to
    // more (which means we need separate weights for each one).
    const int num_data_points = GetNumDataPoints();
    for (const auto& regularizer_and_parameter : regularizers_) {
      const double* estimated_image_data = solver_data.getcontent();
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

    cost_difference = previous_cost - last_iteration_residual_sum_;
    previous_cost = last_iteration_residual_sum_;
    num_iterations_ran++;
    LOG(INFO) << "IRLS Iteration complete (#" << num_iterations_ran << "). "
              << "New loss is " << last_iteration_residual_sum_
              << " with a difference of " << cost_difference << ".";
    if (solver_options_.max_num_irls_iterations > 0 &&
        num_iterations_ran >= solver_options_.max_num_irls_iterations) {
      break;
    }
  }

  const ImageData estimated_image(
      solver_data.getcontent(), GetImageSize(), num_channels);
  return estimated_image;
}

void IrlsMapSolver::NotifyIterationComplete(const double total_residual_sum) {
  last_iteration_residual_sum_ = total_residual_sum;
  if (IsVerbose()) {
    LOG(INFO) << "Callback: residual sum = " << total_residual_sum;
  }
}

}  // namespace super_resolution
