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

  // Loads all frames in the given image directory. This does not technically
  // need to be a video, but rather multiple frames of the same scene.
  void LoadFramesFromDirectory(const std::string& directory_path);

  // Returns the size of the low resolution images. If the size of the images
  // varies, then this will return the size of the first image. If there are no
  // images, the size returned will be (0, 0).
  cv::Size GetImageSize() const;

  // Plays the original video file in a GUI window.
  void PlayOriginalVideo() const;

  // Returns the list of original video frames.
  const std::vector<cv::Mat>& GetFrames() const {
    return video_frames_;
  }

 private:
  // The original (low resolution) video frames.
  std::vector<cv::Mat> video_frames_;
};

}  // namespace super_resolution

#endif  // SRC_VIDEO_VIDEO_LOADER_H_
