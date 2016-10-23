#include "image_model/image_model.h"

#include "image/image_data.h"
#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

void ImageModel::AddDegradationOperator(
    const DegradationOperator& degradation_operator) {

  degradation_operators_.push_back(&degradation_operator);
}

void ImageModel::ApplyModel(
    const ImageData& image_data, const int index) const {

  for (const DegradationOperator* degradation_operator
           : degradation_operators_) {
    degradation_operator->ApplyToImage(image_data, index);
  }
}

}  // namespace super_resolution
