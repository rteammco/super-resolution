// This binary is used to generate low-resolution images from a given
// high-resolution ground truth image.

#include <string>
#include <vector>

#include "data_generator/data_generator.h"
#include "util/macros.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

// Input and output files.
DEFINE_string(input_image, "",
    "Path to the HR image that will be used to generate the LR images.");
DEFINE_string(output_image_dir, "",
    "Path to a directory that will contain all of the generated LR images.");

// Parameters for the low-resolution image generation.
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

  REQUIRE_ARG(FLAGS_input_image);
  REQUIRE_ARG(FLAGS_output_image_dir);

  const cv::Mat image = cv::imread(
      FLAGS_input_image, CV_LOAD_IMAGE_GRAYSCALE);
  super_resolution::DataGenerator data_generator(image);

  data_generator.SetMotionSequence({
      super_resolution::MotionShift(0, 0),
      super_resolution::MotionShift(0, 1),
      super_resolution::MotionShift(2, 0),
      super_resolution::MotionShift(1, 2)});
  data_generator.SetBlurImage(!FLAGS_no_image_blur);
  data_generator.SetNoiseStandardDeviation(FLAGS_noise_standard_deviation);
  std::vector<cv::Mat> low_res_images = data_generator.GenerateLowResImages(
      FLAGS_downsampling_scale, FLAGS_number_of_frames);

  // Save the image to the provided directory.
  for (int i = 0; i < low_res_images.size(); ++i) {
    std::string image_path =
        FLAGS_output_image_dir + "/low_res_" + std::to_string(i) + ".jpg";
    // Convert to 0-255 scale before saving JPG file.
    cv::Mat output_image = low_res_images[i];
    output_image.convertTo(output_image, CV_8UC3, 255.0);
    cv::imwrite(image_path, output_image);
  }

  return EXIT_SUCCESS;
}
