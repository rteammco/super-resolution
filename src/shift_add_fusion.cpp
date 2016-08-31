// Contains code for the shift-add fusion algorithm as explained in "An
// Introduction to Super-Resolution Imaging (2012)".

#include <string>
#include <vector>

#include "data_generator/data_generator.h"
#include "util/macros.h"
#include "util/util.h"
#include "video/video_loader.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

// Input of the LR files.
DEFINE_string(input_image_dir, "",
    "Path to a directory containing the LR images in alphabetical order.");

// Parameters for generating the high-resolution image.
DEFINE_int32(upsampling_scale, 2,
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

  // TODO(richard): Don't hardcode the motion sequence. Eventually estimate the
  // motion automatically.
  std::vector<super_resolution::MotionShift> motion_shifts = {
      super_resolution::MotionShift(0, 0),
      super_resolution::MotionShift(0, 1),
      super_resolution::MotionShift(1, 0),
      super_resolution::MotionShift(1, 1)
  };

  const std::vector<cv::Mat>& frames = video_loader.GetFrames();
  CHECK(motion_shifts.size() == frames.size())
      << "The number of motion estimates must match the number of frames.";

  for (int i = 0; i < frames.size(); ++i) {
    cv::Mat frame = frames[i];
    cv::cvtColor(frame, frame, CV_BGR2GRAY);

    // Add this frame to the SR image.
    for (int x = 0; x < frame.cols; ++x) {
      for (int y = 0; y < frame.rows; ++y) {
        const int hr_x = FLAGS_upsampling_scale * x - motion_shifts[i].dx;
        const int hr_y = FLAGS_upsampling_scale * y - motion_shifts[i].dy;
        if (hr_x < 0 || hr_x >= width || hr_y < 0 || hr_y >= height) {
          continue;
        }
        super_resolved_image.at<uchar>(hr_y, hr_x) = frame.at<uchar>(y, x);
      }
    }
  }

  cv::imshow("disp", super_resolved_image);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
