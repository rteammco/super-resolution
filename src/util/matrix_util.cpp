#include "util/matrix_util.h"

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace util {

void ApplyConvolutionToImage(
    ImageData* image_data, const cv::Mat& kernel, const int border_mode) {

  CHECK_NOTNULL(image_data);

  int num_image_channels = image_data->GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat channel_image = image_data->GetChannelImage(i);
    cv::filter2D(
        channel_image,       // input image
        channel_image,       // output image
        -1,                  // depth of output (-1 = same as input)
        kernel,              // the convolution kernel
        cv::Point(-1, -1),   // anchor kernel at its center
        0,                   // addition to all values (none)
        border_mode);        // border mode (e.g. reflect, pad zeros, etc.)
  }
}

void ThresholdImage(
    cv::Mat image, const double min_value, const double max_value) {

  // Set all values larger than max_value to max_value.
  cv::threshold(image, image, max_value, max_value, cv::THRESH_TRUNC);
  // Set all values smaller than min_value to min_value.
  cv::threshold(image, image, min_value, max_value, cv::THRESH_TOZERO);
}

}  // namespace util
}  // namespace super_resolution
