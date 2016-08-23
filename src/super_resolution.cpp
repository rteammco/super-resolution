#include "util/util.h"
#include "video/super_resolver.h"
#include "video/video_loader.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(video_path, "", "Path to a video file to super resolve.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Super resolution.");

  super_resolution::VideoLoader video_loader;
  video_loader.LoadFramesFromVideo(FLAGS_video_path);
  video_loader.PlayOriginalVideo();

  super_resolution::SuperResolutionOptions options;

  super_resolution::SuperResolver super_resolver(video_loader, options);
  super_resolver.SuperResolve();

  // TODO(richard): the list of algorithm steps (eventually).
  // 1. Verify that the data has 2N frames.
  // 2. Load up all images.
  // 3. Compute SR for the middle image.
  // 4. Evaluate the results.
  // Ultimately, I/O is:
  //  in  => one of my old low-quality videos
  //  out => noticably better quality version of that video

  return EXIT_SUCCESS;
}
