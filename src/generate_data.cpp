// This binary is used to generate low-resolution images from a given
// high-resolution ground truth image.

#include <vector>

#include "data_generator/data_generator.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(image_path, "",
    "Path to the HR image that will be used to generate the LR images.");
DEFINE_bool(no_image_blur, false,
    "Option to not blur the generated LR images.");
DEFINE_int32(noise_standard_deviation, 5,
    "Standard deviation of the noise to be added to the LR images.");
DEFINE_int32(downsampling_scale, 3,
    "The scale by which the HR image will be downsampled and blurred.");
DEFINE_int32(number_of_frames, 4,
    "The number of LR images that will be generated.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv,
      "Generate low-resolution frames from a high-resolution image.");

  CHECK(!FLAGS_image_path.empty()) << "Must provide an image file path.";

  const cv::Mat image = cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_GRAYSCALE);
  super_resolution::DataGenerator data_generator(image);

  cv::imshow("Image", image);
  cv::waitKey(0);

  data_generator.SetMotionSequence({
      super_resolution::MotionShift(0, 0),
      super_resolution::MotionShift(0, 1),
      super_resolution::MotionShift(2, 0),
      super_resolution::MotionShift(1, 2)});
  data_generator.SetBlurImage(!FLAGS_no_image_blur);
  data_generator.SetNoiseStandardDeviation(FLAGS_noise_standard_deviation);
  std::vector<cv::Mat> low_res_images = data_generator.GenerateLowResImages(
      FLAGS_downsampling_scale, FLAGS_number_of_frames);

  cv::Mat vis;
  cv::resize(low_res_images[0], vis, image.size());
  cv::imshow("low res 1", vis);
  cv::imshow("low res 1 norm", low_res_images[0]);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
