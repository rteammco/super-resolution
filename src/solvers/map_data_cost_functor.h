// The super-resolution cost function (defined for Ceres) for the maximum a
// posteriori formulation data term. A different cost function is used for the
// regularization term.

#ifndef SRC_SOLVERS_MAP_DATA_COST_FUNCTOR_H_
#define SRC_SOLVERS_MAP_DATA_COST_FUNCTOR_H_

#include <vector>

#include "image/image_data.h"
#include "solvers/map_cost_processor.h"

#include "ceres/ceres.h"

namespace super_resolution {

// Dynamic numeric cost function implementation.
struct MapDataCostFunctor {
  MapDataCostFunctor(
      const int image_index,
      const int channel_index,
      const int num_pixels,
      const MapCostProcessor& map_cost_processor)
  : image_index_(image_index),
    channel_index_(channel_index),
    num_pixels_(num_pixels),
    map_cost_processor_(map_cost_processor) {}

  // The actual cost function, computes the residuals. "params" is an array of
  // parameter block arrays. There is only one parameter block for this problem
  // which contains the estimated pixel values for the high-resolution image x.
  // Residuals for x are estimated with the image model through the given
  // MapCostProcessor.
  //
  // TODO: do we need to check that all values are between 0 and 1? Or leave it
  // to figure out on its own? Or an extra regularization term?
  bool operator() (double const* const* params, double* residuals) const {
    const double* x = params[0];
    const std::vector<double> computed_residuals =
        map_cost_processor_.ComputeDataTermResiduals(
            image_index_, channel_index_, x);
    for (int i = 0; i < num_pixels_; ++i) {
      // TODO: squared? abs? works as is for now...
      residuals[i] = computed_residuals[i];
    }
    return true;
  }

  // Factory sets up the dynamic cost function with the appropriate number of
  // parameters and residuals.
  static ceres::CostFunction* Create(
      const int image_index,
      const int channel_index,
      const int num_pixels,
      const MapCostProcessor& map_cost_processor) {
    // The cost function takes the estimated HR image (a vector of num_pixels
    // values) and returns a residual for each pixel.
    ceres::DynamicNumericDiffCostFunction<MapDataCostFunctor>* cost_function =
        new ceres::DynamicNumericDiffCostFunction<MapDataCostFunctor>(
            new MapDataCostFunctor(
                image_index, channel_index, num_pixels, map_cost_processor));
    cost_function->AddParameterBlock(num_pixels);
    cost_function->SetNumResiduals(num_pixels);
    return cost_function;
  }

  const int image_index_;
  const int channel_index_;
  const int num_pixels_;
  const MapCostProcessor& map_cost_processor_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_DATA_COST_FUNCTOR_H_
