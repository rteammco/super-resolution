#include "image_model/image_model.h"

#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

void ImageModel::AddDegradationOperator(
    const DegradationOperator& degradation_operator) {

  degradation_operators_.push_back(&degradation_operator);
}

void ImageModel::ApplyModel(cv::Mat* image) const {
  for (const DegradationOperator* degradation_operator
           : degradation_operators_) {
    degradation_operator->ApplyToImage(image);
  }
}

}  // namespace super_resolution
