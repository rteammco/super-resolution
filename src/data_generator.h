#ifndef SRC_DATA_GENERATOR_H_
#define SRC_DATA_GENERATOR_H_

#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

class DataGenerator {
 public:
  explicit DataGenerator(const cv::Mat& image) :
      image_(image),
      blur_image_(true),
      noise_standard_deviation_(5) {}

  // TODO(richard): also include number of images to generate.
  std::vector<cv::Mat> GenerateLowResImages(const int scale) const;

  // TODO(richard): add these functions.
  // SetMotionSequence(...);
  // ;

  // Option setters.
  void SetBlurImage(const bool blur_image) {
    blur_image_ = blur_image;
  }

 private:
  // The original high-resolution image from which the data will be generated.
  const cv::Mat& image_;

  // If true, the image will be blurred during the downsampling process.
  bool blur_image_;

  // The applied noise standard deviation is in pixels.
  int noise_standard_deviation_;
};

}  // namespace super_resolution

#endif  // SRC_DATA_GENERATOR_H_
