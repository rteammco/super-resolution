// The DataGenerator class creates a set of simulated low-resolution image from
// a single high-resolution image. The DataGenerator can be used to test the
// super resolution algorithm against ground truth data.

#ifndef SRC_DATA_GENERATOR_DATA_GENERATOR_H_
#define SRC_DATA_GENERATOR_DATA_GENERATOR_H_

#include <string>
#include <utility>
#include <vector>

#include "data_generator/motion_shift.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

class DataGenerator {
 public:
  // Stores the given image from which the low resolution frames will be
  // generated. Also sets default values.
  explicit DataGenerator(const cv::Mat& high_res_image) :
      high_res_image_(high_res_image),
      blur_image_(true),
      noise_standard_deviation_(5) {}

  // Returns a list of num_images OpenCV images generated from the original
  // high-resolution image. The given scale is the amount by which the original
  // image will be scaled down.
  std::vector<cv::Mat> GenerateLowResImages(
      const int scale, const int num_images) const;

  // Defines the motion shift for each image that is generated. If the given
  // motion sequence is shorter than the number of frames to be generated, the
  // motion sequence will be looped.
  void SetMotionSequence(const std::vector<MotionShift>& motion_shifts) {
    motion_shift_sequence_.SetMotionSequence(motion_shifts);
  }

  // Set the motion sequence from a file.
  void SetMotionSequence(const std::string& file_path) {
    motion_shift_sequence_.LoadSequenceFromFile(file_path);
  }

  // Set to false to disable blurring the image before downsampling.
  void SetBlurImage(const bool blur_image) {
    blur_image_ = blur_image;
  }

  // Set the noise standard deviation (set to 0 to not add any noise). This
  // value is in pixel intensities (0 to 255).
  void SetNoiseStandardDeviation(const int noise_standard_deviation) {
    noise_standard_deviation_ = noise_standard_deviation;
  }

 private:
  // The original high-resolution image from which the data will be generated.
  const cv::Mat& high_res_image_;

  // Contains a list of shift amounts. These define how many pixels each image
  // in the generated sequence will be shifted by from the default position.
  MotionShiftSequence motion_shift_sequence_;

  // If true, the image will be blurred during the downsampling process.
  bool blur_image_;

  // The applied noise standard deviation is in pixels.
  int noise_standard_deviation_;

  // TODO(richard): no-copy allowed.
};

}  // namespace super_resolution

#endif  // SRC_DATA_GENERATOR_DATA_GENERATOR_H_
