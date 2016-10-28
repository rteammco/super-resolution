#include "image_model/image_model.h"

#include <memory>
#include <utility>

#include "image/image_data.h"
#include "image_model/degradation_operator.h"

namespace super_resolution {

void ImageModel::AddDegradationOperator(
    std::unique_ptr<DegradationOperator> degradation_operator) {

  degradation_operators_.push_back(std::move(degradation_operator));
}

void ImageModel::ApplyModel(ImageData* image_data, const int index) const {
  for (const auto& degradation_operator : degradation_operators_) {
    degradation_operator->ApplyToImage(image_data, index);
  }
}

}  // namespace super_resolution
