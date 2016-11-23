#include "solvers/irls_cost_processor.h"

#include <cmath>
#include <memory>
#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "solvers/regularizer.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

IrlsCostProcessor::IrlsCostProcessor(
    const std::vector<ImageData>& low_res_images,
    const ImageModel& image_model,
    const cv::Size& image_size,
    std::unique_ptr<Regularizer> regularizer,
    const double regularization_parameter,
    const std::vector<double>* irls_weights)
    : image_model_(image_model),
      image_size_(image_size),
      regularizer_(std::move(regularizer)),
      regularization_parameter_(regularization_parameter),
      irls_weights_(irls_weights) {

  for (const ImageData& low_res_image : low_res_images) {
    ImageData observation = low_res_image;  // copy
    observation.ResizeImage(image_size_, cv::INTER_AREA);
    observations_.push_back(observation);
  }
}

double IrlsCostProcessor::ComputeDataTermResidual(
    const int image_index,
    const int channel_index,
    const int pixel_index,
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  const ImageData degraded_hr_image(estimated_image_data, image_size_);
  // TODO: assumes a single channel (channel 0).
  const double pixel_value = image_model_.ApplyToPixel(
      degraded_hr_image, image_index, 0, pixel_index);

  return pixel_value - observations_.at(image_index).GetPixelValue(
      channel_index, pixel_index);
}

std::vector<double> IrlsCostProcessor::ComputeRegularizationResiduals(
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  std::vector<double> residuals =
      regularizer_->ComputeResiduals(estimated_image_data);
  for (int i = 0; i < residuals.size(); ++i) {
    const double weight = sqrt(irls_weights_->at(i));
    residuals[i] = regularization_parameter_ * weight * residuals[i];
  }
  return residuals;
}

}  // namespace super_resolution
