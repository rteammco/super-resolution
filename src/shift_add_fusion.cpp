// Contains code for the shift-add fusion algorithm as explained in "An
// Introduction to Super-Resolution Imaging (2012)".

#include <string>

#include "util/macros.h"
#include "util/util.h"
#include "video/video_loader.h"

#include "opencv2/core/core.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

// Input of the LR files.
DEFINE_string(input_image_dir, "",
    "Path to a directory containing the LR images in alphabetical order.");

// Parameters for generating the high-resolution image.
DEFINE_int32(upsampling_scale, 3,
    "The scale by which to up-scale the LR images.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv,
      "A trivial implementation of shift-add fusion.");

  REQUIRE_ARG(FLAGS_input_image_dir);

  super_resolution::VideoLoader video_loader;
  video_loader.LoadFramesFromDirectory(FLAGS_input_image_dir);

  // Create an empty HR image.
  const cv::Size low_res_image_size = video_loader.GetImageSize();
  const int width = FLAGS_upsampling_scale * low_res_image_size.width;
  const int height = FLAGS_upsampling_scale * low_res_image_size.height;
  cv::Mat super_resolved_image = cv::Mat::zeros(width, height, CV_8UC1);

  const std::vector<cv::Mat>& frames = video_loader.GetFrames();
  for (const cv::Mat& frame : frames) {
    // Add this frame to the SR image.
  }

  return EXIT_SUCCESS;
}
