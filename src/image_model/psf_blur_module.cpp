#include "image_model/psf_blur_module.h"

#include "image/image_data.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

PsfBlurModule::PsfBlurModule(const int blur_radius, const double sigma)
    : blur_radius_(blur_radius), sigma_(sigma) {
  CHECK_GE(blur_radius_, 1);
  CHECK_GT(sigma_, 0.0);
  CHECK(blur_radius_ % 2 == 1) << "Blur radius must be an odd number.";

  const cv::Mat kernel_x = cv::getGaussianKernel(blur_radius_, sigma_);
  const cv::Mat kernel_y = cv::getGaussianKernel(blur_radius_, sigma_);
  blur_kernel_ = kernel_x * kernel_y.t();
}

void PsfBlurModule::ApplyToImage(ImageData* image_data, const int index) const {
  const cv::Size kernel_size(blur_radius_, blur_radius_);
  int num_image_channels = image_data->GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat channel_image = image_data->GetChannelImage(i);
    cv::filter2D(
        channel_image,            // input image
        channel_image,            // output image
        -1,                       // depth of output (-1 = same as input)
        blur_kernel_,             // the convolution kernel
        cv::Point(-1, -1),        // anchor kernel at its center
        0,                        // addition to all values (none)
        cv::BORDER_REFLECT_101);  // border mode
  }
}

cv::SparseMat PsfBlurModule::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  return ConvertKernelToOperatorMatrix(blur_kernel_, image_size);
}

}  // namespace super_resolution
