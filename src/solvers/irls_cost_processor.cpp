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

// Minimum residual value for computing IRLS weights, used to avoid division by
// zero.
constexpr double kMinResidualValue = 0.00001;

IrlsCostProcessor::IrlsCostProcessor(
    const std::vector<ImageData>& low_res_images,
    const ImageModel& image_model,
    const cv::Size& image_size,
    std::unique_ptr<Regularizer> regularizer,
    const double regularization_parameter)
    : image_model_(image_model),
      image_size_(image_size),
      regularizer_(std::move(regularizer)),
      regularization_parameter_(regularization_parameter) {

  for (const ImageData& low_res_image : low_res_images) {
    ImageData observation = low_res_image;  // copy
    observation.ResizeImage(image_size_, cv::INTER_NEAREST);
    observations_.push_back(observation);
  }

  // Initialize all IRLS weights to 1.
  // TODO: num_weights also depends on the number of channels in the HR image.
  const int num_weights = image_size_.width * image_size_.height;
  irls_weights_.resize(num_weights);
  std::fill(irls_weights_.begin(), irls_weights_.end(), 1);
}

std::vector<double> IrlsCostProcessor::ComputeDataTermResiduals(
    const int image_index,
    const int channel_index,
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  // Degrade (and re-upsample) the HR estimate with the image model.
  ImageData degraded_hr_image(estimated_image_data, image_size_);
  image_model_.ApplyToImage(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size_, cv::INTER_NEAREST);

  // Compute the residuals by comparing pixel values.
  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    const double residual =
        degraded_hr_image.GetPixelValue(0, i) -
        observations_.at(image_index).GetPixelValue(channel_index, i);
    residuals.push_back(residual);
  }
  return residuals;
}

std::vector<double> IrlsCostProcessor::ComputeDataTermDerivatives(
    const int image_index,
    const double* residuals) const {

  CHECK_NOTNULL(residuals);

  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> derivatives;
  derivatives.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    // TODO:
    // residuals are computed as
    //   r = (UAx - Uy)
    // for LR image y and estimated HR image x. The derivatives are defined as
    //   d = 2*A'U'r
    // where A' and U' are the transposes of A and U, respectively.
    derivatives.push_back(0);
  }

  return derivatives;
}

std::vector<double> IrlsCostProcessor::ComputeRegularizationResiduals(
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  std::vector<double> residuals =
      regularizer_->ComputeResiduals(estimated_image_data);
  for (int i = 0; i < residuals.size(); ++i) {
    const double weight = sqrt(irls_weights_.at(i));
    residuals[i] = regularization_parameter_ * weight * residuals[i];
  }
  return residuals;
}

void IrlsCostProcessor::UpdateIrlsWeights(const double* estimated_image_data) {
  CHECK_NOTNULL(estimated_image_data);

  // TODO: the regularizer is assumed to be L1 norm. Scale appropriately to L*
  // norm based on the regularizer's properties.
  // TODO: also, this assumes a single regularization term. Maybe we can have
  // more than one?
  std::vector<double> regularization_residuals =
      regularizer_->ComputeResiduals(estimated_image_data);
  CHECK_EQ(regularization_residuals.size(), irls_weights_.size())
      << "Number of residuals does not match number of weights.";
  for (int i = 0; i < regularization_residuals.size(); ++i) {
    // TODO: this assumes L1 loss!
    // w = |r|^(p-2)
    irls_weights_[i] =
        1.0 / std::max(kMinResidualValue, regularization_residuals[i]);
  }
}

}  // namespace super_resolution
