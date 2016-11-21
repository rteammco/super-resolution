#include "solvers/map_cost_processor.h"

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

MapCostProcessor::MapCostProcessor(
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

std::vector<double> MapCostProcessor::ComputeDataTermResiduals(
    const int image_index,
    const int channel_index,
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  // Degrade (and re-upsample) the HR estimate with the image model.
  ImageData degraded_hr_image(estimated_image_data, image_size_);
  image_model_.ApplyModel(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size_, cv::INTER_AREA);

  // Compute the residuals by comparing pixel values.
  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    // Pixel value is taken at channel 0 of the degraded_hr_image because the
    // hr images are estimated one channel at a time; hence, there is only one
    // channel in the estimated HR image.
    const double difference =
        degraded_hr_image.GetPixelValue(0, i) -
        observations_.at(image_index).GetPixelValue(channel_index, i);
    residuals.push_back(difference);
  }
  return residuals;
}

std::vector<double> MapCostProcessor::ComputeRegularizationResiduals(
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
