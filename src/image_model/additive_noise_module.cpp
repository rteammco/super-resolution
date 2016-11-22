#include "image_model/additive_noise_module.h"

#include <vector>

#include "image/image_data.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

AdditiveNoiseModule::AdditiveNoiseModule(const double sigma) : sigma_(sigma) {
  CHECK_GT(sigma_, 0.0);
}

void AdditiveNoiseModule::ApplyToImage(
    ImageData* image_data, const int index) const {

  // The image pixels are scaled between 0 and 1, so scale the sigma also.
  const double scaled_sigma = static_cast<double>(sigma_) / 255.0;

  // Add noise separately to each channel.
  const cv::Size image_size = image_data->GetImageSize();
  const int num_image_channels = image_data->GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat noise = cv::Mat(image_size, util::kOpenCvMatrixType);
    cv::randn(noise, 0, scaled_sigma);
    cv::Mat channel_image = image_data->GetChannelImage(i);
    channel_image += noise;
  }
}

double AdditiveNoiseModule::ApplyToPixel(
    const ImageData& image_data,
    const int image_index,
    const int channel_index,
    const int pixel_index) const {

  // TODO: implement.
  return 0.0;
}

}  // namespace super_resolution
