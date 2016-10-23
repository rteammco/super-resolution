#include "image_model/additive_noise_module.h"

#include <vector>

#include "image/image_data.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

AdditiveNoiseModule::AdditiveNoiseModule(const double sigma) : sigma_(sigma) {
  CHECK_GT(sigma_, 0.0);
}

void AdditiveNoiseModule::ApplyToImage(
    const ImageData& image_data, const int index) const {


  // Add noise separately to each channel.
  const cv::Size image_size = image_data.GetImageSize();
  const int image_type = image_data.GetOpenCvType();
  const int num_image_channels = image_data.GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat noise = cv::Mat(image_size, CV_16SC1);
    cv::randn(noise, 0, sigma_);

    cv::Mat noisy_image;
    cv::Mat channel_image = image_data.GetChannel(i);
    channel_image.convertTo(noisy_image, CV_16SC1);
    noisy_image += noise;
    noisy_image.convertTo(channel_image, image_type);
  }
}

}  // namespace super_resolution
