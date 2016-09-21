#include "image_model/motion_module.h"

#include "motion/motion_shift.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

void MotionModule::ApplyToImage(cv::Mat* image, const int index) const {
  const MotionShift& motion_shift =
      motion_shift_sequence_.GetMotionShift(index);
  const cv::Mat shift_kernel = (cv::Mat_<double>(2, 3)
      << 1, 0, motion_shift.dx,
         0, 1, motion_shift.dy);
  cv::warpAffine(*image, *image, shift_kernel, image->size());
}

}  // namespace super_resolution
