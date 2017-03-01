#include "optimization/irls_map_solver.h"

#include <algorithm>
#include <limits>
#include <memory>
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
      solver_options_(solver_options) {}

ImageData IrlsMapSolver::Solve(const ImageData& initial_estimate) {
  const int num_pixels = GetNumPixels();
  const int num_channels = GetNumChannels();
  CHECK_EQ(initial_estimate.GetNumPixels(), GetNumPixels());
  CHECK_EQ(initial_estimate.GetNumChannels(), GetNumChannels());

  // Set up the base objective function, just the data term. The regularization
  // term depends on the weights, so it gets added in the IRLS loop.
  const int num_data_points = GetNumDataPoints();
  const cv::Size image_size = GetImageSize();
  ObjectiveFunction objective_function_data_term_only(num_data_points);
  std::shared_ptr<ObjectiveTerm> data_term(new ObjectiveDataTerm(
      image_model_, observations_, num_channels, image_size));
  objective_function_data_term_only.AddTerm(data_term);

  // Set up the optimization code with ALGLIB.
  // Copy the initial estimate data to the solver's array.
  alglib::real_1d_array solver_data;
  solver_data.setlength(num_data_points);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    double* data_ptr = solver_data.getcontent() + (num_pixels * channel_index);
    const double* channel_ptr = initial_estimate.GetChannelData(channel_index);
    std::copy(channel_ptr, channel_ptr + num_pixels, data_ptr);
  }

  // A vector containing the IRLS weights, one per parameter of the solver
  // system, and one set of weights per regularization term. These weights get
  // reweighted when the solver finishes and the system solves is run again.
  // The effect of reweighting is to allow solving a 1-norm (or arbitrary
  // p-norm) regularizer with least squares. Weights are initialized to 1.
  const int num_regularizers = regularizers_.size();
  std::vector<std::vector<double>> irls_weights(num_regularizers);
  for (int reg_index = 0; reg_index < num_regularizers; ++reg_index) {
    std::vector<double> weights_for_regularizer(num_data_points);
    std::fill(
        weights_for_regularizer.begin(),
        weights_for_regularizer.end(),
        1.0);
    irls_weights[reg_index] = weights_for_regularizer;
  }

  // Scale the option stop criteria parameters based on the number of parameters
  // and strength of the regularizers.
  IrlsMapSolverOptions solver_options_scaled = solver_options_;
  solver_options_scaled.AdjustThresholdsAdaptively(
      GetNumDataPoints(), GetRegularizationParameterSum());

  // Do the IRLS loop. After every iteration, update the IRLS weights and solve
  // again until the change in residual sum is sufficiently low.
  double previous_cost = std::numeric_limits<double>::infinity();
  double cost_difference = solver_options_scaled.cost_decrease_threshold + 1.0;
  int num_iterations_ran = 0;
  while (std::abs(cost_difference) >=
         solver_options_scaled.irls_cost_difference_threshold) {
    // Add the weighted regularization term(s) to the next objective function.
    // This makes a new copy of the objective function for each iteration.
    ObjectiveFunction objective_function = objective_function_data_term_only;
    for (int reg_index = 0; reg_index < num_regularizers; ++reg_index) {
      const auto& regularizer_and_parameter = regularizers_[reg_index];
      std::shared_ptr<ObjectiveTerm> regularization_term(
          new ObjectiveIrlsRegularizationTerm(
              regularizer_and_parameter.first,
              regularizer_and_parameter.second,
              irls_weights[reg_index],
              num_channels,
              image_size));
      objective_function.AddTerm(regularization_term);
    }

    // Run the solver on the reweighted objective function. Solver choice and
    // differentiation method are determined by options.
    double final_cost = 0.0;
    if (solver_options_scaled.use_numerical_differentiation) {
      if (solver_options_scaled.least_squares_solver == CG_SOLVER) {
        final_cost = RunCGSolverNumericalDiff(
            solver_options_scaled, objective_function, &solver_data);
      } else {
        final_cost = RunLBFGSSolverNumericalDiff(
            solver_options_scaled, objective_function, &solver_data);
      }
    } else {
      if (solver_options_scaled.least_squares_solver == CG_SOLVER) {
        final_cost = RunCGSolverAnalyticalDiff(
            solver_options_scaled, objective_function, &solver_data);
      } else {
        final_cost = RunLBFGSSolverAnalyticalDiff(
            solver_options_scaled, objective_function, &solver_data);
      }
    }

    // If there are no regularizers, then no need to continue since the solver
    // already converged and the objective won't change.
    if (num_regularizers == 0) {
      LOG(INFO) << "Least squares done (no regularization terms to reweight).";
      break;
    }

    // Update the IRLS weights.
    // TODO: should this be computed off of the initial estimate? That seems to
    // get better results at the cost of A LOT of extra computational time.
    // TODO: the regularizer is assumed to be L1 norm. Scale appropriately to
    // L* norm based on the regularizer's properties.
    const int num_data_points = GetNumDataPoints();
    for (int reg_index = 0; reg_index < num_regularizers; ++reg_index) {
      const auto& regularizer_and_parameter = regularizers_[reg_index];
      const double* estimated_image_data = solver_data.getcontent();
      const std::vector<double>& regularization_residuals =
          regularizer_and_parameter.first->ApplyToImage(estimated_image_data);
      CHECK_EQ(regularization_residuals.size(), num_data_points)
          << "Number of residuals does not match number of weights.";
      for (int pixel_index = 0; pixel_index < num_data_points; ++pixel_index) {
        // TODO: this assumes L1 loss!
        // w = |r|^(p-2)
        const double residual_value = regularization_residuals[pixel_index];
        irls_weights[reg_index][pixel_index] =
            1.0 / std::max(kMinResidualValue, residual_value);
      }
    }

    cost_difference = previous_cost - final_cost;
    previous_cost = final_cost;
    num_iterations_ran++;
    LOG(INFO) << "IRLS Iteration complete (#" << num_iterations_ran << "). "
              << "New loss is " << final_cost
              << " with a difference of " << cost_difference << ".";
    if (solver_options_scaled.max_num_irls_iterations > 0 &&
        num_iterations_ran >= solver_options_scaled.max_num_irls_iterations) {
      break;
    }
  }

  const ImageData estimated_image(
      solver_data.getcontent(), image_size, num_channels);
  return estimated_image;
}

}  // namespace super_resolution
