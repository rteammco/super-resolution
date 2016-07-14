#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(image_path, "", "The path to an image file.");

int main(int argc, char** argv) {
  google::SetUsageMessage(
      "Super-resolves a video into a higher quality video.");
  google::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  CHECK(!FLAGS_image_path.empty()) << "Must provide an image file path.";

  const cv::Mat image = cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_GRAYSCALE);
  cv::imshow("Image", image);
  cv::waitKey(0);

  // TODO(richard):
  // 1. set up a blur kernel and convolve the image to blur it.
  // 2. downsample the image into multiple frames with different motion offsets.
  const int scale = 2;
  cv::Mat blur_matrix;
  // 1. Verify that the data has 2N frames.
  // 2. Load up all images.
  // 3. Compute SR for the middle image.
  // 4. Evaluate the results.

  // Ultimately, I/O is:
  //  in  => one of my old low-quality videos
  //  out => noticably better quality version of that video

  return EXIT_SUCCESS;
}
