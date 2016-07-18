#ifndef SRC_DATA_GENERATOR_H_
#define SRC_DATA_GENERATOR_H_

#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

class DataGenerator {
 public:
  explicit DataGenerator(const cv::Mat& image) : image_(image) {}

  std::vector<cv::Mat> GenerateLowResImages(const int scale) const;

 private:
  const cv::Mat& image_;
};

}  // namespace super_resolution

#endif  // SRC_DATA_GENERATOR_H_
