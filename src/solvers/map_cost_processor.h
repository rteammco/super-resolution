// This object provides functionality for doing the actual computation of the
// MAP objective function. It handles all of the image processing and
// application of the ImageModel to the high-resolution estimates. This class
// acts as an interface between the OpenCV image processing code and the Ceres
// solver code.

#ifndef SRC_SOLVERS_MAP_COST_PROCESSOR_H_
#define SRC_SOLVERS_MAP_COST_PROCESSOR_H_

#include <memory>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "solvers/regularizer.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class MapCostProcessor {
 public:
  // Stores all of the given parameters. For the given low-resolution images,
  // copies them and stores resized versions to match the high-resolution image
  // size for residual computations at each HR pixel.
  //
  // The given IRLS weights should be compuated before each iteration in the
  // solver and updated in the vector. These weights should be raw weights, NOT
  // the square roots (as that is done in the ComputeRegularizationResiduals
  // function).
  MapCostProcessor(
      const std::vector<ImageData>& low_res_images,
      const ImageModel& image_model,
      const cv::Size& image_size,
      std::unique_ptr<Regularizer> regularizer,
      const double regularization_parameter,
      const std::vector<double>* irls_weights);

  // Compares the given high-resolution image to the low-resolution image of
  // the given index (and channel) by applying the ImageModel to the HR image.
  // The returned values will be the residuals (the difference in pixel
  // intensity) at each pixel of the HR image.
  std::vector<double> ComputeDataTermResiduals(
      const int image_index,
      const int channel_index,
      const double* estimated_image_data) const;

  // Computes the regularization term residuals at each pixel of the given HR
  // image. This operation incorporates the IRLS weights and regularization
  // parameter automatically.
  std::vector<double> ComputeRegularizationResiduals(
      const double* estimated_image_data) const;

 private:
  // Stores the low-resolution images as observations that were scaled up to
  // the size of the high-resolution image.
  std::vector<ImageData> observations_;

  // The image model to degrade the estimated high-resolution images.
  const ImageModel& image_model_;

  // The dimensions (width, height) of the high-resoltion image.
  const cv::Size& image_size_;

  // The regularization term of the cost function, used in the
  // ComputeRegularizationResiduals function. The regularization parameter is
  // non-negative, but may be 0.
  const std::unique_ptr<Regularizer> regularizer_;
  const double regularization_parameter_;

  // The weights for iteratively reweighted least squares, upated by the solver
  // after every iteration. These should be raw weights, NOT modified by the
  // square root or by the regularization parameter.
  const std::vector<double>* irls_weights_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_COST_PROCESSOR_H_
