#include <algorithm>
#include <string>
#include <vector>

#include "image/image_data.h"
#include "image_model/motion_module.h"
#include "motion/registration.h"
#include "util/data_loader.h"
#include "util/util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::ImageData;
using super_resolution::MotionShift;
using super_resolution::MotionShiftSequence;

// The maximum number error allowed for the registration algorithm (distance in
// number of pixels).
constexpr double kTranslationEstimateErrorTolerance = 0.01;

// Path to the test image for testing registration.
static const std::string kTestImagePath =
    super_resolution::util::GetAbsoluteCodePath("test_data/dallas_half.jpg");

// Tests that the peak signal to noise ratio evaluator gives the correct scores.
TEST(Registration, TranslationalRegistration) {
  std::vector<MotionShift> ground_truth_shifts({
    MotionShift(0, 0),
    MotionShift(0, 1),
    MotionShift(2, 0),
    MotionShift(5, 5),
    MotionShift(-5, -1)
  });
  const MotionShiftSequence ground_truth_sequence(ground_truth_shifts);

  // Load the original image and apply motion shifts to it.
  const ImageData original_image =
      super_resolution::util::LoadImage(kTestImagePath);
  const super_resolution::MotionModule motion_module(ground_truth_sequence);
  std::vector<ImageData> shifted_images;
  const int num_motion_shifts = ground_truth_sequence.GetNumMotionShifts();
  for (int i = 0; i < num_motion_shifts; ++i) {
    ImageData shifted_image = original_image;
    motion_module.ApplyToImage(&shifted_image, i);
    shifted_images.push_back(shifted_image);
  }

  // Try to register it and test that the registered results are close to the
  // ground truth.
  const MotionShiftSequence registered_sequence =
      super_resolution::registration::TranslationalRegistration(shifted_images);
  EXPECT_EQ(registered_sequence.GetNumMotionShifts(), num_motion_shifts);
  for (int i = 0; i < num_motion_shifts; ++i) {
    const MotionShift ground_truth_shift = ground_truth_sequence[i];
    const MotionShift estimated_shift = registered_sequence[i];
    EXPECT_NEAR(
        ground_truth_shift.dx,
        estimated_shift.dx,
        kTranslationEstimateErrorTolerance);
    EXPECT_NEAR(
        ground_truth_shift.dy,
        estimated_shift.dy,
        kTranslationEstimateErrorTolerance);
  }
}
