#include "video/video_loader.h"

#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "glog/logging.h"

namespace super_resolution {

void VideoLoader::LoadFramesFromVideo(const std::string& video_path) {
  cv::VideoCapture video_capture(video_path);
  CHECK(video_capture.isOpened())
      << "Failed to open video file: " + video_path;

  LOG(INFO) << "LoadFramesFromVideo()";
}

void VideoLoader::PlayVideo() const {
  LOG(INFO) << "PlayVideo()";
}

}  // namespace super_resolution
