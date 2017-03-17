// This binary is used just to visualize a single image using the image
// visualization tools in the SuperResolution codebase. This binary can display
// regular images as well as false-color representations of hyperspectral
// images.

#include "image/image_data.h"
#include "util/data_loader.h"
#include "util/macros.h"
#include "util/util.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

// The only argument is the path the the image. This can either be a regular
// image file or a hyperspectral configuration file which specifies information
// about a binary ENVI image file that can be loaded at the specified range.
DEFINE_string(image_path, "",
    "The path to an input image file (regular or hyperspectral config).");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Image visualization.");

  REQUIRE_ARG(FLAGS_image_path);

  super_resolution::ImageData image =
      super_resolution::util::LoadImage(FLAGS_image_path);
  super_resolution::util::DisplayImage(image, "Image Visualization");

  return EXIT_SUCCESS;
}
