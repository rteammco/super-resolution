#include "image_model/downsampling_module.h"

#include "image/image_data.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

DownsamplingModule::DownsamplingModule(const double scale) : scale_(scale) {
  CHECK_GE(scale_, 1.0);
}

void DownsamplingModule::ApplyToImage(
    ImageData* image_data, const int index) const {

  const double scale_factor = 1.0 / scale_;
  // Area interpolation method aliases images by dropping pixels.
  image_data->ResizeImage(scale_factor, cv::INTER_AREA);
}

cv::Mat DownsamplingModule::GetOperatorMatrix(
    const int num_pixels, const int index) const {

  // TODO: implement.
  return cv::Mat::eye(num_pixels, num_pixels, CV_64FC1);
}

}  // namespace super_resolution
