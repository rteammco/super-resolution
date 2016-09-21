// Provides a stricture to contain motion estimate and motion shift sequences
// for multiframe low resolution data.

#ifndef SRC_MOTION_MOTION_SHIFT_H_
#define SRC_MOTION_MOTION_SHIFT_H_

#include <string>
#include <vector>

namespace super_resolution {

// Defines the motion (pixel shift) between two images, namely between an image
// and the first image in the frame sequence.
struct MotionShift {
  MotionShift(const double dx, const double dy) : dx(dx), dy(dy) {}
  const double dx;
  const double dy;
};

// Defines an ordered sequence of MotionShift objects. These can be set and
// saved to a file or loaded from a file.
class MotionShiftSequence {
 public:
  // Default constructor.
  MotionShiftSequence() {}

  // Construct with the given motion sequence.
  explicit MotionShiftSequence(const std::vector<MotionShift>& motion_shifts) {
    SetMotionSequence(motion_shifts);
  }

  // Set the motion sequence to the given list.
  void SetMotionSequence(const std::vector<MotionShift>& motion_shifts);

  // Load the motion sequence from a previously-saved file.
  void LoadSequenceFromFile(const std::string& file_path);

  // Save the motion sequence to a text file.
  void SaveSequenceToFile(const std::string& file_path) const;

  // Returns the number of motion shifts.
  int GetNumMotionShifts() const {
    return motion_shifts_.size();
  }

  // Returns the MotionShift at the given index.
  const MotionShift& GetMotionShift(const int index) const;

  // Same as GetMotionShift but with the bracket operator for simplicity.
  const MotionShift& operator[] (const int index) const {
    return GetMotionShift(index);
  }

 private:
  // The list of MotionShift objects.
  std::vector<MotionShift> motion_shifts_;
};

}  // namespace super_resolution

#endif  // SRC_MOTION_MOTION_SHIFT_H_
