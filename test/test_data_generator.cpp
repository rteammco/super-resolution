#include <vector>

#include "motion/motion_shift.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::IsEmpty;

TEST(BasicTest, ExampleTest) {
  const int one = 1;
  EXPECT_EQ(1, 1);
  std::vector<float> vals;
  EXPECT_THAT(vals, IsEmpty());

  super_resolution::MotionShiftSequence motion_shift_sequence;
  std::vector<super_resolution::MotionShift> motion_shifts = {
    super_resolution::MotionShift(1, 1),
    super_resolution::MotionShift(1, 1)
  };
  motion_shift_sequence.SetMotionSequence(motion_shifts);
}
