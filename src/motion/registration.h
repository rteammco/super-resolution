// This file provides everal utility registration functions that perform an
// image registration on some given ImageData images.
//
// TODO: Add support for dense optical flow:
// For dense optical flow, which is a non-linear motion mapping, this is a TODO.
//
// TODO: Add support for non-translational (affine and projective) motion.

#ifndef SRC_MOTION_REGISTRATION_H_
#define SRC_MOTION_REGISTRATION_H_

#include <vector>

#include "image/image_data.h"
#include "motion/motion_shift.h"

namespace super_resolution {
namespace registration {

// Performs translational registration on the given images, with the first
// image in the list as the reference image.
MotionShiftSequence TranslationalRegistration(
    const std::vector<ImageData>& images);

}  // namespace registration
}  // namespace super_resolution

#endif  // SRC_MOTION_REGISTRATION_H_
