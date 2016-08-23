#include "video/video_loader.h"

#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

// The size of a video frame that gets displayed.
static const cv::Size kDisplayFrameSize(1000, 600);

void VideoLoader::LoadFramesFromVideo(const std::string& video_path) {
  cv::VideoCapture video_capture(video_path);
  CHECK(video_capture.isOpened())
      << "Failed to open video file: " + video_path;

  cv::Mat frame;
  while (video_capture.grab()) {
    video_capture.retrieve(frame);
    video_frames_.push_back(frame.clone());
  }

  LOG(INFO) << "Frames successfully loaded from file: " + video_path;
}

void VideoLoader::PlayOriginalVideo() const {
  const std::string window_name = "Original Video";
  cv::namedWindow(window_name);

  for (const cv::Mat& frame : video_frames_) {
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, kDisplayFrameSize);
    cv::imshow(window_name, resized_frame);
  
    if (cv::waitKey(30) >= 0) {
      break;
    }
  }

  cv::destroyWindow(window_name);
}

}  // namespace super_resolution
