#include "image_model/image_model.h"
#include "util/util.h"
#include "video/super_resolver.h"
#include "video/video_loader.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(data_type, "",
    "The type of data to apply super-resolution to. Default is RGB video.");
DEFINE_string(video_path, "", "Path to a video file to super resolve.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Super resolution.");

  if (FLAGS_data_type == "hyperspectral") {
    // super_resolution::hyperspectral::HyperspectralModel model;
    LOG(INFO) << "HS";
  } else {
    // super_resolution::video::VideoModel model;
    LOG(INFO) << "VID";
  }
  return 0;

  super_resolution::video::VideoLoader video_loader;
  video_loader.LoadFramesFromVideo(FLAGS_video_path);
  video_loader.PlayOriginalVideo();

  super_resolution::video::SuperResolutionOptions options;

  super_resolution::video::SuperResolver super_resolver(video_loader, options);
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
