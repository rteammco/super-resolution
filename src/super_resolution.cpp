#include <iostream>
#include <vector>

#include "./data_generator.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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
  super_resolution::DataGenerator data_generator(image);

  cv::imshow("Image", image);
  cv::waitKey(0);

  std::vector<cv::Mat> low_res_images = data_generator.GenerateLowResImages(3);
  cv::Mat vis;
  cv::resize(low_res_images[0], vis, image.size());
  cv::imshow("low res 1", vis);
  cv::waitKey(0);

  // TODO(richard): the list of algorithm steps (eventually).
  // 1. Verify that the data has 2N frames.
  // 2. Load up all images.
  // 3. Compute SR for the middle image.
  // 4. Evaluate the results.
  // Ultimately, I/O is:
  //  in  => one of my old low-quality videos
  //  out => noticably better quality version of that video

  return EXIT_SUCCESS;
}
