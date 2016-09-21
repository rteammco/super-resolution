#include "motion/motion_shift.h"

#include <fstream>
#include <string>
#include <vector>

#include "glog/logging.h"

namespace super_resolution {

void MotionShiftSequence::SetMotionSequence(
    const std::vector<MotionShift>& motion_shifts) {

  motion_shifts_ = std::vector<MotionShift>(motion_shifts);
}

void MotionShiftSequence::LoadSequenceFromFile(const std::string& file_path) {
  std::ifstream fin(file_path);
  CHECK(fin.is_open()) << "Could not open file " << file_path;

  motion_shifts_.clear();
  double dx, dy;
  while (fin >> dx >> dy) {
    motion_shifts_.push_back(MotionShift(dx, dy));
  }
  fin.close();

  LOG(INFO) << "Loaded all " << motion_shifts_.size() << " motion shifts from "
            << file_path;
}

void MotionShiftSequence::SaveSequenceToFile(
    const std::string& file_path) const {

  std::ofstream fout(file_path);
  CHECK(fout.is_open()) << "Could not open file " << file_path;

  for (const MotionShift& motion_shift : motion_shifts_) {
    fout << motion_shift.dx << " " << motion_shift.dy << "\n";
  }
  fout.close();

  LOG(INFO) << "Wrote all " << motion_shifts_.size() << " motion shifts to "
            << file_path;
}

const MotionShift& MotionShiftSequence::GetMotionShift(const int index) const {
    CHECK(index >= 0 && index < motion_shifts_.size())
        << "The given index " << index << " is out of range. "
        << "It must be between 0 and " << (motion_shifts_.size() - 1);

    return motion_shifts_[index];
}

}  // namespace super_resolution
