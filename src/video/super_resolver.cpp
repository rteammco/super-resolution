#include "video/super_resolver.h"

#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace video {

void SuperResolver::SuperResolve() {
  cv::Ptr<cv::superres::SuperResolution> super_resolution =
      cv::superres::createSuperResolution_BTVL1();

  // TODO(richard): OpenCV has several optical flow options:
  //  Farneback
  //  Simple
  //  DualTVL1
  //  Brox
  //  PyrLK
  cv::Ptr<cv::superres::DenseOpticalFlowExt> optical_flow =
      cv::superres::createOptFlow_DualTVL1();

  super_resolution->setOpticalFlow(optical_flow);
  super_resolution->setScale(options_.scale);
  super_resolution->setIterations(options_.num_iterations);
  super_resolution->setTemporalAreaRadius(options_.temporal_radius);
  // TODO(richard): All settings for the super resolution algorithm:
  // double tau - Asymptotic value of steepest descent method.
  // double lambda - Weight parameter to balance data term and smoothness term.
  // double alpha - Parameter of spacial distribution in Bilateral-TV.
  // int btvKernelSize - Kernel size of Bilateral-TV filter.
  // int blurKernelSize - Gaussian blur kernel size.
  // double blurSigma - Gaussian blur sigma.
  // int temporalAreaRadius - Radius of the temporal search area.
  // Ptr<DenseOpticalFlowExt> opticalFlow - Dense optical flow algorithm.

  const std::vector<cv::Mat>& frames = video_loader_.GetFrames();

  cv::Ptr<cv::superres::FrameSource> frame_source =
      cv::superres::createFrameSource_Video("../data/ditchjump.mp4");

  // Skip the first frame because it's usually corrupt.
  cv::Mat first_frame;
  frame_source->nextFrame(first_frame);
  LOG(INFO) << "Dumped first frame.";

  super_resolution->setInput(frame_source);
  LOG(INFO) << "Set input.";


  cv::Mat result_frame;
  for (int i = 1; i < 11; ++i) {
    LOG(INFO) << "Processing next frame...";
    super_resolution->nextFrame(result_frame);
    LOG(INFO) << "Done!";
    cv::Mat before = frames[i];
    cv::resize(before, before, result_frame.size());
    cv::imshow("Before", before);
    cv::imshow("After", result_frame);

    if (cv::waitKey(0) < 0) {
      break;
    }
  }
  cv::destroyAllWindows();

  LOG(INFO) << "SuperResolve()";
}

}  // namespace video
}  // namespace super_resolution
