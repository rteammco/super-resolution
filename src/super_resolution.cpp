#include <vector>

#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/psf_blur_module.h"
#include "util/macros.h"
#include "util/util.h"
#include "video/super_resolver.h"
#include "video/video_loader.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

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

  // Create the forward image model degradation components.
  super_resolution::DownsamplingModule downsampling_module(3);
  super_resolution::PsfBlurModule blur_module(5, 1.0);

  // Create the forward image model: y = DBx
  super_resolution::ImageModel image_model;
  image_model.AddDegradationOperator(blur_module);
  image_model.AddDegradationOperator(downsampling_module);

  const std::vector<cv::Mat>& frames = video_loader.GetFrames();
  for (const cv::Mat& frame : frames) {
    cv::Mat low_res_frame = frame.clone();
    image_model.ApplyModel(&low_res_frame);

    // Display the degradated frame.
    // TODO(richard): remove, and remove OpenCV includes.
    cv::resize(low_res_frame, low_res_frame, frame.size());
    cv::imshow("test", low_res_frame);
    cv::waitKey(0);
  }

  return EXIT_SUCCESS;
}
