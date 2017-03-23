#include "image_model/motion_module.h"

#include <algorithm>
#include <cmath>

#include "image/image_data.h"
#include "motion/motion_shift.h"
#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace {

void ApplyWarpKernel(const cv::Mat& warp_kernel, ImageData* image_data) {
  const cv::Size image_size = image_data->GetImageSize();
  int num_image_channels = image_data->GetNumChannels();
  for (int i = 0; i < num_image_channels; ++i) {
    cv::Mat channel_image = image_data->GetChannelImage(i);
    cv::warpAffine(channel_image, channel_image, warp_kernel, image_size);
  }
}

}  // namespace

void MotionModule::ApplyToImage(ImageData* image_data, const int index) const {
  CHECK_NOTNULL(image_data);

  const MotionShift motion_shift =
      motion_shift_sequence_.GetMotionShift(index);
  const cv::Mat shift_kernel = (cv::Mat_<double>(2, 3)
      << 1, 0, motion_shift.dx,
         0, 1, motion_shift.dy);
  ApplyWarpKernel(shift_kernel, image_data);
}

void MotionModule::ApplyTransposeToImage(
    ImageData* image_data, const int index) const {

  CHECK_NOTNULL(image_data);

  const MotionShift motion_shift =
      motion_shift_sequence_.GetMotionShift(index);
  const cv::Mat reverse_shift_kernel = (cv::Mat_<double>(2, 3)
      << 1, 0, -motion_shift.dx,
         0, 1, -motion_shift.dy);
  ApplyWarpKernel(reverse_shift_kernel, image_data);
}

cv::Mat MotionModule::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_pixels = image_size.width * image_size.height;
  cv::Mat motion_matrix =
      cv::Mat::zeros(num_pixels, num_pixels, util::kOpenCvMatrixType);
  const MotionShift motion_shift = motion_shift_sequence_[index];
  for (int row = 0; row < image_size.height; ++row) {
    for (int col = 0; col < image_size.width; ++col) {
      const int shifted_row = row - motion_shift.dy;
      const int shifted_col = col - motion_shift.dx;
      if (shifted_row >= 0 && shifted_row < image_size.height &&
          shifted_col >= 0 && shifted_col < image_size.width) {
        const int index = row * image_size.width + col;
        const int shifted_index = shifted_row * image_size.width + shifted_col;
        motion_matrix.at<double>(index, shifted_index) = 1;
      }
    }
  }
  return motion_matrix;
}

}  // namespace super_resolution
