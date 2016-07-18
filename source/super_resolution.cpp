#include <iostream>
#include <vector>

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

  // TODO(richard):
  // -1. set up a blur kernel and convolve the image to blur it.-
  // 2. downsample the image into multiple frames with different motion offsets.

  // Blur the image.
  const cv::Mat blur_kernel = (cv::Mat_<double>(3, 3)
    << 1, 2, 1,
       2, 4, 2,
       1, 2, 1) / 16;
  cv::Mat blurred_image;
  cv::filter2D(image, blurred_image, image.depth(), blur_kernel);

  cv::imshow("Image", image);
  cv::waitKey(0);

  // Downsample the image into a set of all the images.
  const int dx[4] = {0, 0, 2, 1};
  const int dy[4] = {0, 1, 0, 2};
  const double noise_std = static_cast<double>(5) / 255;
  const int scale = 3;
  const int num_low_res_images = 4;

  std::vector<cv::Mat> low_res_images;
  const int num_rows = image.rows / scale;
  const int num_cols = image.cols / scale;
  const cv::Size dimensions(num_cols, num_rows);

  for (int i = 0; i < num_low_res_images; ++i) {
    // Shift the image by the given motion amount.
    const cv::Mat shift_kernel = (cv::Mat_<double>(2, 3)
      << 1, 0, dx[i] + 0.5,
         0, 1, dy[i] + 0.5);
    cv::Mat shifted_image;
    cv::warpAffine(
      blurred_image, shifted_image, shift_kernel, blurred_image.size());

    // Downsample the sifted image.
    cv::Mat low_res_image;
    cv::resize(shifted_image, low_res_image, dimensions);

    // Add random noise.
    cv::Mat noise = cv::Mat(dimensions, CV_64F);
    cv::Mat noisy_image;
    normalize(low_res_image, noisy_image, 0.0, 1.0, CV_MINMAX, CV_64F);
    cv::randn(noise, 0, noise_std);
    noisy_image += noise;

    low_res_images.push_back(noisy_image);
  }

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
