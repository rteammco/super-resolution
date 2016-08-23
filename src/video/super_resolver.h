// The SuperResolver provides an interface to apply a super resolution
// algorithm to the given data.

#ifndef SRC_VIDEO_SUPER_RESOLVER_H_
#define SRC_VIDEO_SUPER_RESOLVER_H_

#include "video/video_loader.h"

namespace super_resolution {

struct SuperResolutionOptions {
  // TODO(richard): implement.
};

class SuperResolver {
 public:
  SuperResolver(
      const VideoLoader& video_loader,
      const SuperResolutionOptions& options) {}

  void SuperResolve();

 private:
};

}  // namespace super_resolution

#endif  // SRC_VIDEO_SUPER_RESOLVER_H_
