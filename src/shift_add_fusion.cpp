// Contains code for the shift-add fusion algorithm as explained in "An
// Introduction to Super-Resolution Imaging (2012)".

#include <string>

#include "util/macros.h"
#include "util/util.h"
#include "video/video_loader.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

// Input of the LR files.
DEFINE_string(input_image_dir, "",
    "Path to a directory containing the LR images in alphabetical order.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv,
      "A trivial implementation of shift-add fusion.");

  REQUIRE_ARG(FLAGS_input_image_dir);

  super_resolution::VideoLoader video_loader;
  video_loader.LoadFramesFromDirectory(FLAGS_input_image_dir);

  return EXIT_SUCCESS;
}
