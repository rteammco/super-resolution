#include "motion/registration.h"

#include <utility>
#include <vector>

#include "image/image_data.h"
#include "motion/motion_shift.h"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"  // TODO: Remove!

#include "glog/logging.h"

namespace super_resolution {
namespace registration {
namespace {

// Stores OpenCV keypoints and their associated feature descriptors.
struct KeypointsAndDescriptors {
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keypoints;
};

// Returns a list of keypoints, and their associated feature descriptors,
// detected in the given image.
// TODO: Add a parameter for choosing the feature detection algorithm.
KeypointsAndDescriptors DetectKeypoints(const ImageData& image) {
  // TODO: Don't use channel 0. Instead implement a method in ImageData that
  //       returns a "structure" image (perhaps the average, max, or median
  //       pixel intensities across all channels). It should work for
  //       hyperspectral images as well.
  cv::Mat detection_image;
  image.GetChannelImage(0).convertTo(detection_image, CV_8U, 255);

  KeypointsAndDescriptors keypoints_and_descriptors;
  cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
  detector->detectAndCompute(
      detection_image,
      cv::noArray(),
      keypoints_and_descriptors.keypoints,
      keypoints_and_descriptors.descriptors);

  if (keypoints_and_descriptors.keypoints.empty()) {
    LOG(WARNING) << "No keypoints detected for the given image.";
  }

  return keypoints_and_descriptors;
}

// Computes pairwise matches between the two given keypoint sets.
std::vector<std::pair<cv::Point2f, cv::Point2f>> FindMatchingFeatures(
    const KeypointsAndDescriptors& keypoints_and_descriptors_1,
    const KeypointsAndDescriptors& keypoints_and_descriptors_2) {

  // If there are no features available for one of the images, returns an empty
  // set of matches.
  if (keypoints_and_descriptors_1.descriptors.empty() ||
      keypoints_and_descriptors_2.descriptors.empty()) {
    return {};  // Default empty list.
  }

  // Need to convert to CV_32F so that the OpenCV Flann-based matcher can work.
  cv::Mat descriptors_1;
  keypoints_and_descriptors_1.descriptors.convertTo(descriptors_1, CV_32F);
  cv::Mat descriptors_2;
  keypoints_and_descriptors_2.descriptors.convertTo(descriptors_2, CV_32F);

  // Run the Flann-based matcher.
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> feature_matches;
  matcher.match(descriptors_1, descriptors_2, feature_matches);

  // Build a list of keypoint match pairs.
  std::vector<std::pair<cv::Point2f, cv::Point2f>> all_keypoint_matches;
  for (const cv::DMatch& match : feature_matches) {
    cv::Point2f pixel_loc_1 =
        keypoints_and_descriptors_1.keypoints[match.queryIdx].pt;
    cv::Point2f pixel_loc_2 =
        keypoints_and_descriptors_2.keypoints[match.trainIdx].pt;
    all_keypoint_matches.push_back(std::make_pair(pixel_loc_1, pixel_loc_2));
  }

  // TODO: Filter out the keypoint matches and only keep the "good" ones.

  return all_keypoint_matches;
}

}  // namespace

MotionShiftSequence TranslationalRegistration(
    const std::vector<ImageData>& images) {

  std::vector<MotionShift> motion_shifts;
  const int num_images = images.size();
  if (num_images == 1) {
    motion_shifts.push_back(MotionShift(0, 0));
  } else if (num_images > 1) {
    const KeypointsAndDescriptors& image_0_keypoints =
        DetectKeypoints(images[0]);
    for (int i = 1; i < num_images; ++i) {
      const KeypointsAndDescriptors& image_i_keypoints =
          DetectKeypoints(images[i]);
      const auto& keypoint_matches = FindMatchingFeatures(
          image_0_keypoints, image_i_keypoints);

      // TODO: Remove! Replace this with the motion computation (homography).
      cv::Mat vis;
      cv::hconcat(
          images[0].GetVisualizationImage(),
          images[i].GetVisualizationImage(),
          vis);
      const cv::Scalar vis_line_color(0, 255, 0);
      for (const auto& keypoint_pair : keypoint_matches) {
        cv::line(
          vis,
          keypoint_pair.first,
          keypoint_pair.second + cv::Point2f(images[i].GetImageSize().width, 0),
          vis_line_color,
          1,  // Line thickness.
          CV_AA);
      }
      cv::imshow("Keypoint Matches", vis);
      cv::waitKey(0);
      // ----
    }
  }
  return MotionShiftSequence(motion_shifts);
}

}  // namespace registration
}  // namespace super_resolution
