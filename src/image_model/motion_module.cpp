#include "image_model/motion_module.h"

#include "image/image_data.h"
#include "motion/motion_shift.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

void MotionModule::ApplyToImage(ImageData* image_data, const int index) const {
  const MotionShift& motion_shift =
      motion_shift_sequence_.GetMotionShift(index);
  const cv::Mat shift_kernel = (cv::Mat_<double>(2, 3)
      << 1, 0, motion_shift.dx,
         0, 1, motion_shift.dy);
  const cv::Size image_size = image_data->GetImageSize();
  int num_image_channels = image_data->GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat channel_image = image_data->GetChannelImage(i);
    cv::warpAffine(channel_image, channel_image, shift_kernel, image_size);
  }
}

cv::Mat MotionModule::GetOperationMatrix(
    const cv::Size& image_size, const int index) const {

  // TODO: implement.
  const int num_pixels = image_size.width * image_size.height;
  return cv::Mat::zeros(num_pixels, num_pixels, util::kOpenCvMatrixType);
}

}  // namespace super_resolution
