#include "image_model/motion_module.h"

#include <algorithm>
#include <cmath>

#include "image/image_data.h"
#include "motion/motion_shift.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

void MotionModule::ApplyToImage(ImageData* image_data, const int index) const {
  const MotionShift motion_shift =
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

int MotionModule::GetPixelPatchRadius() const {
  int radius = 0;
  const int num_motion_shifts = motion_shift_sequence_.GetNumMotionShifts();
  for (int i = 0; i < num_motion_shifts; ++i) {
    const MotionShift motion_shift = motion_shift_sequence_[i];
    const int dx = static_cast<int>(ceil(std::abs(motion_shift.dx)));
    const int dy = static_cast<int>(ceil(std::abs(motion_shift.dx)));
    radius = std::max(radius, std::max(dx, dy));
  }
  return radius;
}

cv::Mat MotionModule::ApplyToPatch(
    const cv::Mat& patch,
    const int image_index,
    const int channel_index,
    const int pixel_index) const {

  const int patch_radius = GetPixelPatchRadius();
  const cv::Size patch_size = patch.size();
  CHECK_GE((patch_size.width / 2), patch_radius)
      << "Patch is too small to apply motion module.";
  CHECK_GE((patch_size.height / 2), patch_radius)
      << "Patch is too small to apply motion module.";

  // Apply the motion warp to the patch.
  const MotionShift motion_shift = motion_shift_sequence_[image_index];
  const cv::Mat shift_kernel = (cv::Mat_<double>(2, 3)
      << 1, 0, motion_shift.dx,
         0, 1, motion_shift.dy);
  cv::Mat degraded_patch;
  cv::warpAffine(patch, degraded_patch, shift_kernel, patch_size);

  // Crop out only the relevant part of the patch.
  const int degraded_patch_width = patch_size.width - (patch_radius * 2);
  const int degraded_patch_height = patch_size.height - (patch_radius * 2);
  return degraded_patch(cv::Rect(
      patch_radius, patch_radius, degraded_patch_width, degraded_patch_height));

//  // TODO: implement.
//  LOG(WARNING) << "Method not implemented. Returning empty patch.";
//  const cv::Mat empty_patch;
//  return empty_patch;
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
