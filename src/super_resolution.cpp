#include <utility>
#include <vector>

#include "hyperspectral/data_loader.h"
#include "image_model/additive_noise_module.h"
#include "image_model/blur_module.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "optimization/map_solver.h"
#include "util/macros.h"
#include "util/util.h"
#include "video/super_resolver.h"
#include "video/video_loader.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

using super_resolution::DegradationOperator;

DEFINE_string(data_type, "",
    "The type of data to apply super-resolution to. Default is RGB video.");
DEFINE_string(video_path, "", "Path to a video file to super resolve.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Super resolution.");

  if (FLAGS_data_type == "hyperspectral") {
    // super_resolution::hyperspectral::HyperspectralModel model;
    LOG(INFO) << "HS";
  } else {
    // super_resolution::video::VideoModel model;
    LOG(INFO) << "VID";
  }

  REQUIRE_ARG(FLAGS_video_path);

  super_resolution::video::VideoLoader video_loader;
  video_loader.LoadFramesFromVideo(FLAGS_video_path);
  video_loader.PlayOriginalVideo();

  // Create the motion estimates.
  super_resolution::MotionShiftSequence motion_shift_sequence({
      super_resolution::MotionShift(10, 3),
      super_resolution::MotionShift(-10, 15),
      super_resolution::MotionShift(0, 0),
      super_resolution::MotionShift(5, 10),
      super_resolution::MotionShift(-8, -10),
      super_resolution::MotionShift(3, -15)
  });

  // Create the forward image model degradation components.
  const super_resolution::DownsamplingModule downsampling_module(
      3, cv::Size(100, 100));  // TODO: use the real image size.
  const super_resolution::MotionModule motion_module(motion_shift_sequence);
  const super_resolution::BlurModule blur_module(5, 1.0);
  const super_resolution::AdditiveNoiseModule noise_module(5.0);

  // Create the forward image model: y = DBx + n
  super_resolution::ImageModel image_model(3);
  image_model.AddDegradationOperator(motion_module);
  image_model.AddDegradationOperator(blur_module);
  image_model.AddDegradationOperator(downsampling_module);
  image_model.AddDegradationOperator(noise_module);

  const std::vector<cv::Mat>& frames = video_loader.GetFrames();
  for (int i = 0; i < frames.size(); ++i) {
    cv::Mat low_res_frame = frames[i].clone();
    // image_model.ApplyToImage(&low_res_frame, i); // TODO: fix

    // Display the degradated frame.
    // TODO: remove, and remove OpenCV includes.
    cv::resize(low_res_frame, low_res_frame, frames[i].size());
    cv::imshow("test", low_res_frame);
    cv::waitKey(0);
  }

  return EXIT_SUCCESS;
}
