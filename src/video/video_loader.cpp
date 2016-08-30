#include "video/video_loader.h"

#include <string>
#include <vector>

#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "glog/logging.h"

namespace super_resolution {

// The size of a video frame that gets displayed.
// TODO(richard): Move this to an optional parameter somewhere.
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

void VideoLoader::LoadFramesFromDirectory(const std::string& directory_path) {
  std::vector<std::string> files_in_directory =
      util::ListFilesInDirectory(directory_path);
  for (const std::string& file_name : files_in_directory) {
    const std::string file_path = directory_path + "/" + file_name;
    cv::Mat frame = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);
    // Skip invalid images.
    if (frame.cols == 0 || frame.rows == 0) {
      LOG(WARNING) << "Skipped file " << file_path
                   << ": could not read image. "
                   << "Make sure it is a valid image type.";
      continue;
    }
    video_frames_.push_back(frame);
  }
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
