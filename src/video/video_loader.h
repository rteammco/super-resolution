// TODO(richard): comments.

#ifndef SRC_VIDEO_VIDEO_LOADER_H_
#define SRC_VIDEO_VIDEO_LOADER_H_

#include <string>

namespace super_resolution {

class VideoLoader {
 public:
  void LoadFramesFromVideo(const std::string& video_path);

  void PlayVideo() const;

 private:
};

}  // namespace super_resolution

#endif  // SRC_VIDEO_VIDEO_LOADER_H_
