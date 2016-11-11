#include "image_model/degradation_operator.h"

#include "util/util.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

cv::Mat DegradationOperator::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_pixels = image_size.width * image_size.height;
  return cv::Mat::eye(num_pixels, num_pixels, util::kOpenCvMatrixType);
}

}  // namespace super_resolution
