#include <iostream>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(
    images_dir, "", "A directory containing the video frame images.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "Hello, world" << std::endl;

  if (!FLAGS_images_dir.empty()) {
    std::cout << "WIN" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
  }

  // 1. Verify that the data has 2N frames.
  // 2. Load up all images.
  // 3. Compute SR for the middle image.
  // 4. Evaluate the results.

  // Ultimately, I/O is:
  //  in  => one of my old low-quality videos
  //  out => noticably better quality version of that video

  return EXIT_SUCCESS;
}
