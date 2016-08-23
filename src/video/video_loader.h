// TODO(richard): comments.

#ifndef SRC_VIDEO_VIDEO_LOADER_H_
#define SRC_VIDEO_VIDEO_LOADER_H_

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

namespace super_resolution {

class VideoLoader {
 public:
  void LoadFramesFromVideo(const std::string& video_path);

  void PlayOriginalVideo() const;

 private:
  std::vector<const cv::Mat> video_frames_;
};

}  // namespace super_resolution

#endif  // SRC_VIDEO_VIDEO_LOADER_H_
