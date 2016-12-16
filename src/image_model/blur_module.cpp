#include "image_model/blur_module.h"

#include "image/image_data.h"
#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

BlurModule::BlurModule(const int blur_radius, const double sigma)
    : blur_radius_(blur_radius) {

  CHECK_GE(blur_radius, 1);
  CHECK_GT(sigma, 0.0);
  CHECK(blur_radius % 2 == 1) << "Blur radius must be an odd number.";

  const cv::Mat kernel_x = cv::getGaussianKernel(blur_radius, sigma);
  const cv::Mat kernel_y = cv::getGaussianKernel(blur_radius, sigma);
  blur_kernel_ = kernel_x * kernel_y.t();
}

void BlurModule::ApplyToImage(ImageData* image_data, const int index) const {
  CHECK_NOTNULL(image_data);
  util::ApplyConvolutionToImage(image_data, blur_kernel_);
}

void BlurModule::ApplyTransposeToImage(
    ImageData* image_data, const int index) const {

  CHECK_NOTNULL(image_data);

  util::ApplyConvolutionToImage(image_data, blur_kernel_.t());
}

cv::Mat BlurModule::ApplyToPatch(
    const cv::Mat& patch,
    const int image_index,
    const int channel_index,
    const int pixel_index) const {

  // TODO: implement.
  LOG(WARNING) << "Method not implemented. Returning empty patch.";
  const cv::Mat empty_patch;
  return empty_patch;
}

cv::Mat BlurModule::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  return ConvertKernelToOperatorMatrix(blur_kernel_, image_size);
}

}  // namespace super_resolution
