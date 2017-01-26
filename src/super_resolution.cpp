#include <utility>
#include <vector>

#include "hyperspectral/hyperspectral_data_loader.h"
#include "image/image_data.h"
#include "image_model/additive_noise_module.h"
#include "image_model/blur_module.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "optimization/map_solver.h"
#include "util/data_loader.h"
#include "util/macros.h"
#include "util/util.h"
#include "video/super_resolver.h"
#include "video/video_loader.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

using super_resolution::ImageData;

// Input images (required):
DEFINE_string(data_path, "",
    "Path to an input file or directory to super resolve.");

// Image model parameters:
DEFINE_int32(upsampling_scale, 2,
    "The amount by which to super-resolve the image(s).");
DEFINE_int32(blur_radius, 3,
    "The radius of the blur kernel. Set to 0 to inactivate blurring.");
DEFINE_double(blur_sigma, 1.0,
    "The sigma value of the Gaussian blur. Set to 0 to inactivate blurring.");
DEFINE_string(motion_sequence_path, "",
    "Path to a file containing the motion shifts for each image.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Super resolution.");

  REQUIRE_ARG(FLAGS_data_path);

  std::vector<ImageData> images = super_resolution::util::LoadImages(
      FLAGS_data_path);

  // Create the forward image model.
  super_resolution::ImageModel image_model(FLAGS_upsampling_scale);

  // TODO: fix this issue.
  //if (!FLAGS_motion_sequence_path.empty()) {
  //  super_resolution::MotionShiftSequence motion_shift_sequence;
  //  motion_shift_sequence.LoadSequenceFromFile(FLAGS_motion_sequence_path);
  //  const super_resolution::MotionModule motion_module(motion_shift_sequence);
  //  image_model.AddDegradationOperator(motion_module);
  //}

  //if (FLAGS_blur_radius > 0 && FLAGS_blur_sigma > 0) {
    const super_resolution::BlurModule blur_module(
        FLAGS_blur_radius, FLAGS_blur_sigma);
    image_model.AddDegradationOperator(blur_module);
  //}

  const super_resolution::DownsamplingModule downsampling_module(
      FLAGS_upsampling_scale);
  image_model.AddDegradationOperator(downsampling_module);

  LOG(INFO) << "Running super-resolution on " << images.size() << " images.";
  // TODO: super-resolve.
  ImageData x = images[0];
  image_model.ApplyToImage(&x, 0);
  cv::imshow("Image", x.GetVisualizationImage());
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
