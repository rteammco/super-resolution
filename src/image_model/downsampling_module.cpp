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

  const double scale_ratio = 1.0 / scale_;
  const int num_image_channels = image_data->GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat channel_image = image_data->GetChannel(i);
    cv::resize(
        channel_image,     // Source image.
        channel_image,     // Dest image (overwrite the old image).
        cv::Size(0, 0),    // Size is set to 0, so it will use the ratio.
        scale_ratio,       // Scaling ratio in the x asix (0 < r <= 1).
        scale_ratio,       // Scaling ratio in the y axis.
        cv::INTER_AREA);   // Area method aliases images by dropping pixels.
  }
}

}  // namespace super_resolution
