// The MapSolver is an implementation framework for the maximum a posteriori
// formulation. This is an abstract class, since the MAP formulation can be
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

// The available solvers to use for least squares minimization.
enum LeastSquaresSolver {
  CG_SOLVER,    // Conjugate gradient solver.
  LBFGS_SOLVER  // Limited-memory BFGS solver.
};

// Options for the solver. Set/update these as needed for subclasses of
// MapSolver.
struct MapSolverOptions {
  MapSolverOptions() {}  // Required for making a const instance.

  // Adaptively adjusts all of the termination criteria thresholds based on the
  // number of parameters and the regularization parameter sum. These factors
  // contribute to the total cost of the objective function, so scaled
  // threshold values may be desirable for quicker convergence.
  //
  // NOTE: The base thresholds must be set appropriately beforehand for this
  // adjustment to work.
  virtual void AdjustThresholdsAdaptively(
      const int num_parameters, const double regularization_parameter_sum);

  // Neatly prints out all options used for the user.
  virtual void PrintSolverOptions() const;

  // Which solver to use.
  LeastSquaresSolver least_squares_solver = CG_SOLVER;

  // The number of corrections for approximating the Hessian for an LBFGS
  // iteration. ALGLIB recommends 3 <= num_lbfgs_hessian_corrections <= 7.
  //
  // Only applicable if using LBFGS_SOLVER.
  int num_lbfgs_hessian_corrections = 5;

  // Maximum number of solver iterations. 0 for infinite.
  int max_num_solver_iterations = 50;

  // Thresholds for stopping the solver if:
  //   The norm of the gradient is smaller than this.
  double gradient_norm_threshold = 1.0e-6;
  //   The change (decrease) in the cost is smaller than this.
  double cost_decrease_threshold = 1.0e-6;
  //   The change in the norm of the parameter vector is smaller than this.
  double parameter_variation_threshold = 1.0e-6;

  // Optional parameters for numerical differentiation. Use for testing
  // analytical differentiation only. Numerical differentiation is very slow
  // but gives near-perfect estimates of the gradient. It is not feasible for
  // larger data sets.
  bool use_numerical_differentiation = false;
  double numerical_differentiation_step = 1.0e-6;
};

class MapSolver : public Solver {
 public:
  // Constructor is the same as Solver constructor but also takes the
  // low-resolution images as input.
  MapSolver(
      const ImageModel& image_model,
      const std::vector<ImageData>& low_res_images,
      const bool print_solver_output = true);

  // Adds a regularizer term to the objective function.
  virtual void AddRegularizer(
      std::shared_ptr<Regularizer> regularizer,
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

  // Returns the sum of all regularization parameters.
  double GetRegularizationParameterSum() const;

 protected:
  // All regularization terms and their respective regularization parameters to
  // be applied in the cost function.
  std::vector<std::pair<std::shared_ptr<Regularizer>, double>> regularizers_;

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
