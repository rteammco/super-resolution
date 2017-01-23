// The MapSolver is an implementation framework for the maximum a posteriori
// formulation. This is a pure virtual class, since the MAP formulation can be
// solved by different strategies, such as IRLS (iteratively reweighted least
// squares). See subclasses for the specific solver strategy implementations.

#ifndef SRC_OPTIMIZATION_MAP_SOLVER_H_
#define SRC_OPTIMIZATION_MAP_SOLVER_H_

#include <memory>
#include <utility>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "optimization/regularizer.h"
#include "optimization/solver.h"

namespace super_resolution {

// TODO: remove this.
enum RegularizationMethod {
  TOTAL_VARIATION
};

struct MapSolverOptions {
  MapSolverOptions() {}

  // TODO: fill in the options as needed.
  // Using temporary (old) options to avoid compile errors.
  double regularization_parameter = 0.0;
  RegularizationMethod regularization_method = TOTAL_VARIATION;
  bool use_numerical_differentiation = false;
};

class MapSolver : public Solver {
 public:
  // Constructor is the same as Solver constructor but also takes the
  // MapSolverOptions struct.
  MapSolver(
      const MapSolverOptions& solver_options,
      const ImageModel& image_model,
      const std::vector<ImageData>& low_res_images,
      const bool print_solver_output = true);

  // Adds a regularizer term to the objective function.
  void AddRegularizer(  // TODO: virtual or not?
      std::unique_ptr<Regularizer> regularizer,
      const double regularization_parameter);

  // Returns the number of pixels in a single image channel. This is NOT the
  // total number of pixels in the entire image.
  int GetNumPixels() const {
    return image_size_.width * image_size_.height;
  }

  // Returns the spatial size (width, height) of the image.
  cv::Size GetImageSize() const {
    return image_size_;
  }

  // Returns the number of channels (spectral bands) in each observation image.
  // This will be the number of channels in the high-resolution estimate.
  int GetNumChannels() const {
    return num_channels_;
  }

  // Returns the number of observations (low-resolution images) in the solver
  // system.
  int GetNumImages() const {
    return observations_.size();
  }

  // Returns the number of data points, which is the total number of pixels in
  // an image across all channels.
  int GetNumDataPoints() const {
    return GetNumPixels() * GetNumChannels();
  }

 protected:
  const MapSolverOptions solver_options_;

  // All regularization terms and their respective regularization parameters to
  // be applied in the cost function.
  std::vector<std::pair<std::unique_ptr<Regularizer>, double>> regularizers_;

  // The observed LR images scaled up to the HR image size for use in the cost
  // function.
  std::vector<ImageData> observations_;

 private:
  // This is the size of the HR image that is being estimated.
  cv::Size image_size_;

  // This is the number of channels in each image. Guaranteed to be consistent
  // accross all observations.
  int num_channels_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_MAP_SOLVER_H_
