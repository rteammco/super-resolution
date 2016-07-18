#include "./data_generator.h"

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace super_resolution {

std::vector<cv::Mat> DataGenerator::GenerateLowResImages(
    const int scale) const {

  // Blur the image if the blur flag is enabled.
  cv::Mat blurred_image;
  if (blur_image_) {
    // Make kernel_size so that it is odd even if scale is even.
    const int kernel_size = (scale % 2 == 1) ? scale : scale - 1;
    cv::GaussianBlur(
        image_, blurred_image, cv::Size(kernel_size, kernel_size), kernel_size);
  } else {
    blurred_image = image_;
  }

  // Downsample the image into a set of all the images.
  const int dx[4] = {0, 0, 2, 1};
  const int dy[4] = {0, 1, 0, 2};
  const double noise_std = static_cast<double>(5) / 255;
  const int num_low_res_images = 4;

  std::vector<cv::Mat> low_res_images;
  const int num_rows = image_.rows / scale;
  const int num_cols = image_.cols / scale;
  const cv::Size dimensions(num_cols, num_rows);

  for (int i = 0; i < num_low_res_images; ++i) {
    // Shift the image by the given motion amount.
    const cv::Mat shift_kernel = (cv::Mat_<double>(2, 3)
      << 1, 0, dx[i] + 0.5,
         0, 1, dy[i] + 0.5);
    cv::Mat shifted_image;
    cv::warpAffine(
      blurred_image, shifted_image, shift_kernel, blurred_image.size());

    // Downsample the sifted image.
    cv::Mat low_res_image;
    cv::resize(shifted_image, low_res_image, dimensions);

    // Add random noise.
    cv::Mat noise = cv::Mat(dimensions, CV_64F);
    cv::Mat noisy_image;
    normalize(low_res_image, noisy_image, 0.0, 1.0, CV_MINMAX, CV_64F);
    cv::randn(noise, 0, noise_std);
    noisy_image += noise;

    low_res_images.push_back(noisy_image);
  }

  return low_res_images;
}

}  // namespace super_resolution
