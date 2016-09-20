// The SuperResolver provides an interface to apply a super resolution
// algorithm to the given data.

#ifndef SRC_VIDEO_SUPER_RESOLVER_H_
#define SRC_VIDEO_SUPER_RESOLVER_H_

#include <string>

#include "video/video_loader.h"

namespace super_resolution {
namespace video {

// All possible options for the super resolution algorithm.
struct SuperResolutionOptions {
  std::string super_resolution_algorithm = "";
  std::string optical_flow_algorithm = "";

  int scale = 2;
  int num_iterations = 1;
  int temporal_radius = 3;

  double blur_kernel_size = 3;
  double blur_sigma = 3;

  bool use_gpu = false;
};

class SuperResolver {
 public:
  SuperResolver(
      const VideoLoader& video_loader,
      const SuperResolutionOptions& options) :
  video_loader_(video_loader),
  options_(options) {}

  // Apply the super resolution algorithm to the given data (in the given
  // VideoLoader). All parameters will be set according to the given
  // SuperResolutionOptions.
  void SuperResolve();

 private:
  // The VideoLoader object containing the low-resolution video frames.
  const VideoLoader& video_loader_;

  const SuperResolutionOptions& options_;
};

}  // namespace video
}  // namespace super_resolution

#endif  // SRC_VIDEO_SUPER_RESOLVER_H_
