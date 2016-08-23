#include "video/super_resolver.h"

#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"

#include "glog/logging.h"

namespace super_resolution {

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
  // TODO(richard): All settings for the super resolution algorithm:
  // int scale - Scale factor.
  // int iterations - Iteration count.
  // double tau - Asymptotic value of steepest descent method.
  // double lambda - Weight parameter to balance data term and smoothness term.
  // double alpha - Parameter of spacial distribution in Bilateral-TV.
  // int btvKernelSize - Kernel size of Bilateral-TV filter.
  // int blurKernelSize - Gaussian blur kernel size.
  // double blurSigma - Gaussian blur sigma.
  // int temporalAreaRadius - Radius of the temporal search area.
  // Ptr<DenseOpticalFlowExt> opticalFlow - Dense optical flow algorithm.

  LOG(INFO) << "SuperResolve()";
}

}  // namespace super_resolution
