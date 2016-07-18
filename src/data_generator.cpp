#include "./data_generator.h"

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace super_resolution {

std::vector<cv::Mat> DataGenerator::GenerateLowResImages(
    const int scale, const int num_images) const {

  // Blur the image if the blur flag is enabled.
  cv::Mat blurred_image;
  if (blur_image_) {
    // Make kernel_size so that it is odd even if scale is even.
    const int kernel_size = (scale % 2 == 1) ? scale : scale - 1;
    cv::GaussianBlur(
        high_res_image_,
        blurred_image,
        cv::Size(kernel_size, kernel_size),
        kernel_size);
  } else {
    blurred_image = high_res_image_;
  }

  const cv::Size low_res_dimensions(
      high_res_image_.cols / scale, high_res_image_.rows / scale);
  const double noise_std = static_cast<double>(noise_standard_deviation_) / 255;
  const int num_motion_shifts = motion_shifts_.size();

  std::vector<cv::Mat> low_res_images;
  for (int i = 0; i < num_images; ++i) {
    // Shift the image by the given motion amounts (if any are given).
    cv::Mat shifted_image;
    if (num_motion_shifts > 0) {
      const MotionShift& motion_shift = motion_shifts_[i % num_motion_shifts];
      const cv::Mat shift_kernel = (cv::Mat_<double>(2, 3)
        << 1, 0, motion_shift.dx + 0.5,
           0, 1, motion_shift.dy + 0.5);
      cv::warpAffine(
          blurred_image, shifted_image, shift_kernel, blurred_image.size());
    } else {
      shifted_image = blurred_image;
    }

    // Downsample the sifted image.
    cv::Mat low_res_image;
    cv::resize(shifted_image, low_res_image, low_res_dimensions);

    // Add random noise.
    cv::Mat noise = cv::Mat(low_res_dimensions, CV_64F);
    cv::Mat noisy_image;
    normalize(low_res_image, noisy_image, 0.0, 1.0, CV_MINMAX, CV_64F);
    cv::randn(noise, 0, noise_std);
    noisy_image += noise;

    low_res_images.push_back(noisy_image);
  }

  return low_res_images;
}

void DataGenerator::SetMotionSequence(
    const std::vector<MotionShift>& motion_shifts) {

  motion_shifts_ = std::vector<MotionShift>(motion_shifts);
}

}  // namespace super_resolution
