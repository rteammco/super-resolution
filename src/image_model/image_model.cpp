#include "image_model/image_model.h"

#include <memory>
#include <utility>

#include "image/image_data.h"
#include "image_model/additive_noise_module.h"
#include "image_model/blur_module.h"
#include "image_model/degradation_operator.h"
#include "image_model/downsampling_module.h"
#include "image_model/motion_module.h"

#include "glog/logging.h"

namespace super_resolution {

ImageModel ImageModel::CreateImageModel(
    const ImageModelParameters& parameters) {

  ImageModel image_model(parameters.scale);

  // Add motion if a motion sequence or file is provided.
  if (!parameters.motion_sequence_path.empty() ||
       parameters.motion_sequence.GetNumMotionShifts() > 0) {
    std::shared_ptr<MotionModule> motion_module;
    if (parameters.motion_sequence.GetNumMotionShifts() > 0) {
      // If motion sequence was provided:
      motion_module = std::shared_ptr<MotionModule>(
          new MotionModule(parameters.motion_sequence));
    } else {
      // If file name was provided:
      MotionShiftSequence motion_shift_sequence;
      motion_shift_sequence.LoadSequenceFromFile(
          parameters.motion_sequence_path);
      motion_module = std::shared_ptr<MotionModule>(
          new MotionModule(motion_shift_sequence));
    }
    image_model.AddDegradationOperator(motion_module);
  }

  // Add blur if the blur parameters are non-zero.
  if (parameters.blur_radius > 0 && parameters.blur_sigma > 0.0) {
    std::shared_ptr<BlurModule> blur_module(
        new BlurModule(parameters.blur_radius, parameters.blur_sigma));
    image_model.AddDegradationOperator(blur_module);
  }

  // Add the downsampling operator.
  std::shared_ptr<DownsamplingModule> downsampling_module(
      new DownsamplingModule(parameters.scale));
  image_model.AddDegradationOperator(downsampling_module);

  // Add noise if the noise sigma is positive.
  if (parameters.noise_sigma > 0.0) {
    std::shared_ptr<AdditiveNoiseModule> noise_module(
        new AdditiveNoiseModule(parameters.noise_sigma));
    image_model.AddDegradationOperator(noise_module);
  }

  return image_model;
}

ImageModel::ImageModel(const int downsampling_scale)
    : downsampling_scale_(downsampling_scale) {

  CHECK_GE(downsampling_scale_, 1)
      << "Downsampling scale must be at least 1. 1 means no downsampling.";
}

void ImageModel::AddDegradationOperator(
    std::shared_ptr<DegradationOperator> degradation_operator) {

  degradation_operators_.push_back(degradation_operator);
}

ImageData ImageModel::ApplyToImage(
    const ImageData& image_data, const int index) const {

  ImageData degraded_image = image_data;
  for (const auto& degradation_operator : degradation_operators_) {
    degradation_operator->ApplyToImage(&degraded_image, index);
  }
  return degraded_image;
}

void ImageModel::ApplyToImage(ImageData* image_data, const int index) const {
  CHECK_NOTNULL(image_data);
  for (const auto& degradation_operator : degradation_operators_) {
    degradation_operator->ApplyToImage(image_data, index);
  }
}

void ImageModel::ApplyTransposeToImage(
    ImageData* image_data, const int index) const {

  CHECK_NOTNULL(image_data);
  const int num_degradation_operators = degradation_operators_.size();
  for (int i = num_degradation_operators - 1; i >= 0; --i) {
    degradation_operators_[i]->ApplyTransposeToImage(image_data, index);
  }
}

cv::Mat ImageModel::GetModelMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_operators = degradation_operators_.size();
  CHECK_GT(num_operators, 0)
      << "Cannot build a model matrix with no degradation operators.";

  cv::Mat model_matrix =
      degradation_operators_[0]->GetOperatorMatrix(image_size, index);
  for (int i = 1; i < num_operators; ++i) {
    const cv::Mat next_matrix =
        degradation_operators_[i]->GetOperatorMatrix(image_size, index);
    model_matrix = next_matrix * model_matrix;
  }
  return model_matrix;
}

}  // namespace super_resolution
