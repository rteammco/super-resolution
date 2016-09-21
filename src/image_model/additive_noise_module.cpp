#include "image_model/additive_noise_module.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

AdditiveNoiseModule::AdditiveNoiseModule(const double sigma) : sigma_(sigma) {
  CHECK_GT(sigma_, 0.0);
}

void AdditiveNoiseModule::ApplyToImage(cv::Mat* image) const {
  // TODO(richard): This method is only for 3-channel images.
  cv::Mat noise = cv::Mat(image->size(), CV_16SC3);
  cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(sigma_));

  cv::Mat noisy_image;
  image->convertTo(noisy_image, CV_16SC3);
  noisy_image += noise;
  noisy_image.convertTo(*image, image->type());
}

}  // namespace super_resolution
