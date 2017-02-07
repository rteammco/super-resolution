// This motion degradation module simply applies a translational transformation
// on each image in the frame sequence based on the given MotionShiftSequence.

#ifndef SRC_IMAGE_MODEL_MOTION_MODULE_H_
#define SRC_IMAGE_MODEL_MOTION_MODULE_H_

#include "image/image_data.h"
#include "image_model/degradation_operator.h"
#include "motion/motion_shift.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class MotionModule : public DegradationOperator {
 public:
  // The given MotionShiftSequence should provide motion information for each
  // image in the frame sequence.
  explicit MotionModule(const MotionShiftSequence motion_shift_sequence)
      : motion_shift_sequence_(motion_shift_sequence) {}

  virtual void ApplyToImage(ImageData* image_data, const int index) const;

  virtual void ApplyTransposeToImage(
      ImageData* image_data, const int index) const;

  virtual cv::Mat GetOperatorMatrix(
      const cv::Size& image_size, const int index) const;

 private:
  const MotionShiftSequence motion_shift_sequence_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_MOTION_MODULE_H_
