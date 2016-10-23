// This binary is used to generate low-resolution images from a given
// high-resolution ground truth image.
#include <string>
#include <vector>

#include "image/image_data.h"
#include "image_model/additive_noise_module.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "image_model/psf_blur_module.h"
#include "motion/motion_shift.h"
#include "util/macros.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

// Required input and output files.
DEFINE_string(input_image, "",
    "Path to the HR image that will be used to generate the LR images.");
DEFINE_string(output_image_dir, "",
    "Path to a directory that will contain all of the generated LR images.");

// Motion estimate file I/O parameters.
DEFINE_string(input_motion_sequence, "",
    "Path to a text file containing a simulated motion sequence.");

// Parameters for the low-resolution image generation.
DEFINE_int32(blur_radius, 0,
    "The radius of the Gaussian blur kernel. If 0, no blur will be added.");
DEFINE_double(blur_sigma, 0.0,
    "The sigma of the Gaussian blur kernel. If 0, no blur will be added.");
DEFINE_double(noise_sigma, 0.0,
    "Standard deviation of the additive noise. If 0, no noise will be added.");
DEFINE_int32(downsampling_scale, 2,
    "The scale by which the HR image will be downsampled.");
DEFINE_int32(number_of_frames, 4,
    "The number of LR images that will be generated.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv,
      "Generate low-resolution frames from a high-resolution image.");

  REQUIRE_ARG(FLAGS_input_image);
  REQUIRE_ARG(FLAGS_output_image_dir);

  const cv::Mat image = cv::imread(FLAGS_input_image, CV_LOAD_IMAGE_GRAYSCALE);
  super_resolution::ImageData image_data(image);

  // Set up a motion sequence from a file if the user specified one.
  super_resolution::MotionShiftSequence motion_shift_sequence;
  if (!FLAGS_input_motion_sequence.empty()) {
    motion_shift_sequence.LoadSequenceFromFile(FLAGS_input_motion_sequence);
  }

  // Set up the ImageModel with all the parameters specified by the user. This
  // model will be used to generate the degradated images.
  super_resolution::ImageModel image_model;

  // Add motion.
  super_resolution::MotionModule motion_module(motion_shift_sequence);
  image_model.AddDegradationOperator(motion_module);

  // Add blur if the parameters are specified.
  if (FLAGS_blur_radius > 0 && FLAGS_blur_sigma > 0) {
    super_resolution::PsfBlurModule blur_module(
        FLAGS_blur_radius, FLAGS_blur_sigma);
    image_model.AddDegradationOperator(blur_module);
  }

  // Add downsampling.
  super_resolution::DownsamplingModule downsampling_module(
      FLAGS_downsampling_scale);
  image_model.AddDegradationOperator(downsampling_module);

  // Add additive noise if the parameter was specified.
  if (FLAGS_noise_sigma > 0) {
    super_resolution::AdditiveNoiseModule noise_module(FLAGS_noise_sigma);
    image_model.AddDegradationOperator(noise_module);
  }

  for (int i = 0; i < FLAGS_number_of_frames; ++i) {
    super_resolution::ImageData low_res_frame = image_data;
    image_model.ApplyModel(low_res_frame, i);
    // Write the file.
    std::string image_path =
        FLAGS_output_image_dir + "/low_res_" + std::to_string(i) + ".jpg";
    cv::imwrite(image_path, low_res_frame.GetChannel(0));
  }

  return EXIT_SUCCESS;
}
