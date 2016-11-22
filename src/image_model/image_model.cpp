#include "image_model/image_model.h"

#include <memory>
#include <utility>

#include "image/image_data.h"
#include "image_model/degradation_operator.h"

#include "glog/logging.h"

namespace super_resolution {

void ImageModel::AddDegradationOperator(
    std::unique_ptr<DegradationOperator> degradation_operator) {

  degradation_operators_.push_back(std::move(degradation_operator));
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
  for (const auto& degradation_operator : degradation_operators_) {
    degradation_operator->ApplyToImage(image_data, index);
  }
}

double ImageModel::ApplyToPixel(
    const ImageData& image_data,
    const int image_index,
    const int channel_index,
    const int pixel_index) {

  return 0.0;
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
    // TODO: multiplication doesn't work with this SparseMat representation.
    model_matrix = next_matrix * model_matrix;
  }
  return model_matrix;
}

}  // namespace super_resolution
