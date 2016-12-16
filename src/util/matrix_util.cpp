#include "util/matrix_util.h"

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {
namespace util {

void ApplyConvolutionToImage(
    ImageData* image_data, const cv::Mat& kernel, const int border_mode) {

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

}  // namespace util
}  // namespace super_resolution
