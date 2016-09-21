#include "image_model/additive_noise_module.h"

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

AdditiveNoiseModule::AdditiveNoiseModule(const double sigma) : sigma_(sigma) {
  CHECK_GT(sigma_, 0.0);
}

void AdditiveNoiseModule::ApplyToImage(cv::Mat* image, const int index) const {
  // Split the image up into individual channels.
  const int num_image_channels = image->channels();
  std::vector<cv::Mat> channels(num_image_channels);
  cv::split(*image, channels);

  // Add noise separately to each channel.
  const cv::Size image_size = image->size();
  const int image_type = image->type();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat noise = cv::Mat(image_size, CV_16SC1);
    cv::randn(noise, 0, sigma_);

    cv::Mat noisy_image;
    channels[i].convertTo(noisy_image, CV_16SC1);
    noisy_image += noise;
    noisy_image.convertTo(channels[i], image_type);
  }

  // Put the noisy channels back together.
  cv::merge(channels, *image);
}

}  // namespace super_resolution
