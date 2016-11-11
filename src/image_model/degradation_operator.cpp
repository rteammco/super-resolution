#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

cv::Mat DegradationOperator::GetOperatorMatrix(
    const int num_pixels, const int index) const {

  // TODO: the image type really needs to be defined as a global constant
  // somewhere else.
  return cv::Mat::eye(num_pixels, num_pixels, CV_64FC1);
}

}  // namespace super_resolution
