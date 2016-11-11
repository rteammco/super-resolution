#include "image_model/degradation_operator.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

cv::Mat DegradationOperator::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_pixels = image_size.width * image_size.height;
  // TODO: the image type really needs to be defined as a global constant
  // somewhere else.
  return cv::Mat::eye(num_pixels, num_pixels, CV_64FC1);
}

}  // namespace super_resolution
