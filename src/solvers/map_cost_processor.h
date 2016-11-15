// This object provides functionality for doing the actual computation of the
// MAP objective function. It handles all of the image processing and
// application of the ImageModel to the high-resolution estimates. This class
// acts as an interface between the OpenCV image processing code and the Ceres
// solver code.

#ifndef SRC_SOLVERS_MAP_COST_PROCESSOR_H_
#define SRC_SOLVERS_MAP_COST_PROCESSOR_H_

#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class MapCostProcessor {
 public:
  MapCostProcessor(const std::vector<ImageData>& low_res_images,
                   const ImageModel& image_model,
                   const cv::Size& high_res_image_size);

  // Compares the given high-resolution image to the low-resolution image of
  // the given index (and channel) by applying the ImageModel to the HR image.
  // The returned values will be the residuals (the difference in pixel
  // intensity) at each pixel of the HR image.
  //
  // TODO: high_res_image_data should be cached so it doesn't need to be
  // re-created (converted to double) more than once per iteration.
  std::vector<double> ComputeDataTermResiduals(
      const int image_index,
      const int channel_index,
      const std::vector<double>& high_res_image_data) const;

  // Computes the regularization term residuals at each pixel of the given HR
  // image.
  //
  // TODO: the regularization operator should be given as a parameter to the
  // MapCostProcessor object.
  std::vector<double> ComputeregularizationResiduals(
      const int channel_index,
      const std::vector<double>& high_res_image_data) const;

 private:
  // Stores the low-resolution images as observations that were scaled up to
  // the size of the high-resolution image.
  std::vector<ImageData> observations_;

  // The image model to degrade the estimated high-resolution images.
  const ImageModel& image_model_;

  // The dimensions (width, height) of the high-resoltion image.
  const cv::Size& high_res_image_size_;
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_MAP_COST_PROCESSOR_H_
