#include "image_model/downsampling_module.h"

#include <cmath>

#include "image/image_data.h"
#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

DownsamplingModule::DownsamplingModule(
    const int scale, const cv::Size& image_size)
    : scale_(scale), image_size_(image_size) {

  CHECK_GE(scale_, 1);
}

void DownsamplingModule::ApplyToImage(
    ImageData* image_data, const int index) const {

  CHECK_NOTNULL(image_data);

  const double scale_factor = 1.0 / static_cast<double>(scale_);
  // Area interpolation method aliases images by dropping pixels.
  image_data->ResizeImage(scale_factor, cv::INTER_NEAREST);
}

void DownsamplingModule::ApplyTransposeToImage(
    ImageData* image_data, const int index) const {

  CHECK_NOTNULL(image_data);

  // The transpose of downsampling is trivial upsampling to size * scale_. We
  // will use non-interpolating upsampling, so the pixels will be mapped to the
  // higher resolution with zeros padded between them. This is defined in
  // ImageData::UpsampleImage().
  image_data->UpsampleImage(scale_);
}

// TODO: implementation only works for scale = 2 and 3x3 patches for now.
cv::Mat DownsamplingModule::ApplyToPatch(
    const cv::Mat& patch,
    const int image_index,
    const int channel_index,
    const int pixel_index) const {

  // TODO: actual required patch size is only scale_ x scale_, where the pixel
  // in question is in the bottom right corner.
  const int required_patch_size = 2 * scale_ - 1;
  const cv::Size patch_size = patch.size();
  CHECK_GE(patch_size.width, required_patch_size);
  CHECK_GE(patch_size.height, required_patch_size);

  // TODO: remove all the commented LOGGING code once everything is debugged.

  // LOG(INFO) << "Given patch size: " << patch_size.width;
  // LOG(INFO) << "Required patch size: " << required_patch_size;

  // Compute the (x, y) coordinate of the top-left corner of this patch in the
  // image and use it to compute the downsampling offset (e.g. if 2x
  // downsampling, we sample every 2 pixels - so the offset is either 0 or 1).
  const int pixel_x = pixel_index % image_size_.width;
  const int pixel_y = pixel_index / image_size_.width;

  // LOG(INFO) << "Pixel pos (x, y): " << pixel_x << ", " << pixel_y;

  const int patch_left_x = pixel_x - patch_size.width / 2;
  const int patch_top_y = pixel_y - patch_size.height / 2;

  // LOG(INFO) << "Top left pos (x, y): "
  //           << patch_left_x << ", " << patch_top_y;

  const int patch_offset_x = abs(patch_left_x) % scale_;
  const int patch_offset_y = abs(patch_top_y) % scale_;

  // LOG(INFO) << "Patch offset (x, y): "
  //           << patch_offset_x << ", " << patch_offset_y;

  const int patch_radius = GetPixelPatchRadius();
  const int cropped_patch_width = patch_size.width - patch_radius;
  const int cropped_patch_height = patch_size.height - patch_radius;

  // LOG(INFO) << "Cropped patch size: " << cropped_patch_width;

  // Crop out the part of the patch that we're going to downsample based on the
  // offsets.
  cv::Mat cropped_patch = patch(cv::Rect(
      patch_offset_x,
      patch_offset_y,
      cropped_patch_width,
      cropped_patch_height));

  // LOG(INFO) << "Cropped patch:";
  // LOG(INFO) << cropped_patch;

  // Perform standard downsampling on the offset-cropped patch.
  const cv::Size cropped_patch_size = cropped_patch.size();
  const cv::Size new_size(
      cropped_patch_size.width / scale_,
      cropped_patch_size.height / scale_);

  // LOG(INFO) << "New patch size: " << new_size.width;

  cv::Mat resized_patch;
  cv::resize(cropped_patch, resized_patch, new_size, 0, 0, cv::INTER_NEAREST);

  // LOG(INFO) << "New patch: " << resized_patch;

  return resized_patch;
}

cv::Mat DownsamplingModule::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_high_res_pixels = image_size.width * image_size.height;
  const int num_low_res_pixels = num_high_res_pixels / (scale_ * scale_);
  cv::Mat downsampling_matrix = cv::Mat::zeros(
      num_low_res_pixels, num_high_res_pixels, util::kOpenCvMatrixType);

  int next_row = 0;
  for (int row = 0; row < image_size.height; ++row) {
    if (row % scale_ != 0) {
      continue;
    }
    for (int col = 0; col < image_size.width; ++col) {
      if (col % scale_ != 0) {
        continue;
      }
      const int index = row * image_size.width + col;
      downsampling_matrix.at<double>(next_row, index) = 1;
      next_row++;
    }
  }
  return downsampling_matrix;
}

}  // namespace super_resolution
