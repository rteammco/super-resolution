// The super-resolution cost function (defined for Ceres) for the maximum a
// posteriori formulation.

#ifndef SRC_SOLVERS_MAP_COST_FUNCTION_H_
#define SRC_SOLVERS_MAP_COST_FUNCTION_H_

#include <vector>

#include "image/image_data.h"

#include "ceres/ceres.h"

namespace super_resolution {

struct MapCostFunction {
  MapCostFunction(
      const std::vector<ImageData>* observation_estimates,
      const int image_index,
      const int channel_index,
      const int pixel_index)
  : observation_estimates_(observation_estimates),
    image_index_(image_index),
    channel_index_(channel_index),
    pixel_index_(pixel_index) {}

  template <typename T>
  bool operator() (const T* const ingored_value, T* residual) const {
    // NOTE: the input parameter is estimate of X. Ignore it, because we need
    // to use the the estimated Y_i instead which is computed before every
    // iteration.

    // Get the estimated pixel value.
    const double estimated_pixel_value =
        observation_estimates_->at(image_index_).GetPixelValue(
            channel_index_, pixel_index_);

    residual[0] = T(0.0);
    return true;
  }

  static ceres::CostFunction* Create(
      const std::vector<ImageData>* observation_estimates,
      const int image_index,
      const int channel_index,
      const int pixel_index) {
    // The cost function takes one value - a pixel intensity - and returns the
    // residual between that pixel intensity and the expected observation.
    return (new ceres::AutoDiffCostFunction<MapCostFunction, 1, 1>(
        new MapCostFunction(
            observation_estimates,
            image_index,
            channel_index,
            pixel_index)));
  }

  const std::vector<ImageData>* observation_estimates_;
  const int image_index_;
  const int channel_index_;
  const int pixel_index_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_COST_FUNCTION_H_
