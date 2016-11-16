// The super-resolution cost function (defined for Ceres) for the maximum a
// posteriori formulation.

#ifndef SRC_SOLVERS_MAP_COST_FUNCTOR_H_
#define SRC_SOLVERS_MAP_COST_FUNCTOR_H_

#include <iostream> // TODO: remove
#include <vector>

#include "image/image_data.h"
#include "solvers/map_cost_processor.h"

#include "ceres/ceres.h"

namespace super_resolution {

// Dynamic numeric cost function implementation.
// TODO: the dynamic part.
struct MapCostFunctor {
  MapCostFunctor(
      const int image_index,
      const int channel_index,
      const int num_pixels,
      const MapCostProcessor& map_cost_processor)
  : image_index_(image_index),
    channel_index_(channel_index),
    num_pixels_(num_pixels),
    map_cost_processor_(map_cost_processor) {}

  bool operator() (const double* const x, double* residuals) const {
    // Check that all values are between 0 and 1, otherwise it's outside of the
    // trust region.
    //for (int i = 0; i < num_pixels_; ++i) {
    //  if (x[i] < 0.0 || x[i] > 1.0) {
    //    return false;
    //  }
    //}

    const std::vector<double> computed_residuals =
        map_cost_processor_.ComputeDataTermResiduals(
            image_index_, channel_index_, x);
    //std::cout << "Residuals = ";
    for (int i = 0; i < num_pixels_; ++i) {
      residuals[i] = computed_residuals[i];  // TODO: squared? abs?
      //std::cout << computed_residuals[i] << ", ";
    }
    //std::cout << std::endl;
    //std::cout << "END OF ITERATION" << std::endl;
    return true;
  }

  static ceres::CostFunction* Create(
      const int image_index,
      const int channel_index,
      const int num_pixels,
      const MapCostProcessor& map_cost_processor) {
    // The cost function takes one value - a pixel intensity - and returns the
    // residual between that pixel intensity and the expected observation.
    return (new ceres::NumericDiffCostFunction<
        MapCostFunctor, ceres::CENTRAL, 16, 16>(new MapCostFunctor(
            image_index, channel_index, num_pixels, map_cost_processor)));
  }

  const int image_index_;
  const int channel_index_;
  const int num_pixels_;
  const MapCostProcessor& map_cost_processor_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_COST_FUNCTOR_H_
