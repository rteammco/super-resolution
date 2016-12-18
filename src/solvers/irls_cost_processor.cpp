#include "solvers/irls_cost_processor.h"

#include <algorithm>
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

  // TODO: it would be nice if this could just compute the residuals and
  // derivatives at the same time.

  ImageData upgraded_residual_image(residuals, image_size_);
  const int scale = image_model_.GetDownsamplingScale();
  const cv::Size lr_image_size(
      image_size_.width / scale, image_size_.height / scale);
  upgraded_residual_image.ResizeImage(lr_image_size);
  image_model_.ApplyTransposeToImage(&upgraded_residual_image, image_index);

  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> derivatives;
  derivatives.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    const double derivative =
        2 * upgraded_residual_image.GetPixelValue(0, i);  // Only 1 channel.
    derivatives.push_back(derivative);
  }

  return derivatives;
}

std::vector<double> IrlsCostProcessor::ComputeRegularizationResiduals(
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  std::vector<double> residuals =
      regularizer_->ApplyToImage(estimated_image_data);
  for (int i = 0; i < residuals.size(); ++i) {
    const double weight = sqrt(irls_weights_.at(i));
    residuals[i] = regularization_parameter_ * weight * residuals[i];
  }
  return residuals;
}

std::vector<double> IrlsCostProcessor::ComputeRegularizationDerivatives(
    const double* estimated_image_data) const {

  CHECK_NOTNULL(estimated_image_data);

  // TODO: it would be nice if this could just compute the residuals and
  // derivatives at the same time.

  // We are computing the derivative as
  //   2r*W'W*g(x)*d(g(x))
  // so we just need the regularizer to return g(x) and d(g(x)) with respect to
  // every parameter in x.
  const std::vector<double> regularizer_values =
      regularizer_->ApplyToImage(estimated_image_data);
  
  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> partial_const_terms;
  for (int i = 0; i < num_pixels; ++i) {
    // Each derivative is multiplied by
    //   2 * lambda * w^2 * reg_i
    // where 2 comes from the squared norm (L2) term,
    // lambda is the regularization parameter,
    // w^2 is the squared weight (since the weights are square-rooted in the
    //   residual computation, the raw weight is used here),
    // and reg_i is the value of the regularization term at pixel i.
    // These constants are multiplied with the partial derivatives at each
    // pixel w.r.t. all other pixels, which are computed specifically based on
    // the derivative of the regularizer function.
    partial_const_terms.push_back(
        2 *
        regularization_parameter_ *
        irls_weights_[i] *
        regularizer_values[i]);
  }
  const std::vector<double> derivatives = regularizer_->GetDerivatives(
      estimated_image_data, partial_const_terms);
  return derivatives;
}

void IrlsCostProcessor::UpdateIrlsWeights(const double* estimated_image_data) {
  CHECK_NOTNULL(estimated_image_data);

  // TODO: the regularizer is assumed to be L1 norm. Scale appropriately to L*
  // norm based on the regularizer's properties.
  // TODO: also, this assumes a single regularization term. Maybe we can have
  // more than one?
  std::vector<double> regularization_residuals =
      regularizer_->ApplyToImage(estimated_image_data);
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
