// The VideoLoader class handles all file I/O and converts videos into
// individual video frames. Load low-resolution videos with this class and
// apply super resolution on the individual frames.

#ifndef SRC_VIDEO_VIDEO_LOADER_H_
#define SRC_VIDEO_VIDEO_LOADER_H_

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

class VideoLoader {
 public:
  // Loads all frames of the given video file. The given path must be a valid
  // video file supported by OpenCV.
  void LoadFramesFromVideo(const std::string& video_path);

  // Plays the original video file in a GUI window.
  void PlayOriginalVideo() const;

  // Returns the list of original video frames.
  const std::vector<const cv::Mat>& GetFrames() const {
    return video_frames_;
  }

 private:
  // The original (low resolution) video frames.
  std::vector<const cv::Mat> video_frames_;
};

}  // namespace super_resolution

#endif  // SRC_VIDEO_VIDEO_LOADER_H_
