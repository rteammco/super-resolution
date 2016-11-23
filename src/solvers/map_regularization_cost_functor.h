// The regularization term of the super-resolution cost function (defined for
// Ceres) for the maximum a posteriori formulation. A separate cost function is
// used for the data fidelity term.

#ifndef SRC_SOLVERS_MAP_REGULARIZATION_COST_FUNCTOR_H_
#define SRC_SOLVERS_MAP_REGULARIZATION_COST_FUNCTOR_H_

#include <algorithm>
#include <vector>

#include "solvers/irls_cost_processor.h"

#include "ceres/ceres.h"

namespace super_resolution {

struct MapRegularizationCostFunctor {
  MapRegularizationCostFunctor(
      const IrlsCostProcessor& irls_cost_processor)
  : irls_cost_processor_(irls_cost_processor) {}

  // The actual cost function, computes the residuals. "params" is an array of
  // parameter block arrays. There is only one parameter block for this problem
  // which contains the estimated pixel values for the high-resolution image x.
  // Residuals for x are estimated with the through the given
  // IrlsCostProcessor.
  bool operator() (double const* const* params, double* residuals) const {
    // TODO: this should be just one residual.
    const std::vector<double> computed_residuals =
        irls_cost_processor_.ComputeRegularizationResiduals(params[0]);
    std::copy(computed_residuals.begin(), computed_residuals.end(), residuals);
    return true;
  }

  // Factory sets up the dynamic cost function with the appropriate number of
  // parameters and residuals.
  static ceres::CostFunction* Create(
      const int num_pixels,
      const IrlsCostProcessor& irls_cost_processor) {
    // The cost function takes the estimated HR image (a vector of num_pixels
    // values) and returns a residual for each pixel.
    ceres::DynamicNumericDiffCostFunction<MapRegularizationCostFunctor>*
    cost_function =
        new ceres::DynamicNumericDiffCostFunction<MapRegularizationCostFunctor>(
            new MapRegularizationCostFunctor(irls_cost_processor));
    cost_function->AddParameterBlock(num_pixels);
    cost_function->SetNumResiduals(num_pixels);
    return cost_function;
  }

  const IrlsCostProcessor& irls_cost_processor_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_REGULARIZATION_COST_FUNCTOR_H_
