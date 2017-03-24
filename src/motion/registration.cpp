#include "motion/registration.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "image/image_data.h"
#include "motion/motion_shift.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"  // TODO: Remove!

#include "glog/logging.h"

namespace super_resolution {
namespace registration {
namespace {

constexpr double kFlannDistanceScalingFactor = 5.0;
constexpr double kFlannDistanceThreshold = 0.04;
constexpr double kRansacReprojectionThreshold = 0.1;

using KeypointPairing =
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>;

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

KeypointPairing ApplyRANSAC(const KeypointPairing& unfiltered_matches) {
  if (unfiltered_matches.first.size() < 3) {
    LOG(WARNING) << "Cannot apply RANSAC with less than 3 keypoint matches ("
                 << unfiltered_matches.first.size() << " given).";
    return unfiltered_matches;
  }

  CHECK_EQ(unfiltered_matches.first.size(), unfiltered_matches.second.size())
      << "Imbalanced keypoint pairs. "
      << "Number of matched keypoints must be the same across both images.";

  std::vector<unsigned char> inliers_mask;
  cv::findHomography(
      unfiltered_matches.first,
      unfiltered_matches.second,
      CV_RANSAC,
      kRansacReprojectionThreshold,
      inliers_mask);

  KeypointPairing filtered_matches;
	for (int i = 0; i < inliers_mask.size(); ++i) {
    if (inliers_mask[i] != static_cast<unsigned char>(0)) {
      filtered_matches.first.push_back(unfiltered_matches.first[i]);
      filtered_matches.second.push_back(unfiltered_matches.second[i]);
    }
  }
  LOG(INFO) << "Filtered from " << unfiltered_matches.first.size()
            << " to " << filtered_matches.first.size() << " matches.";
  return filtered_matches;
}

// Computes pairwise keypoint matches between the two given feature descriptor
// sets. This does does not guarantee ideal matches. Further filtering, such as
// RANSAC, may be necessary.
KeypointPairing FindMatchingFeatures(
    const KeypointsAndDescriptors& keypoints_and_descriptors_1,
    const KeypointsAndDescriptors& keypoints_and_descriptors_2) {

  // If there are no features available for one of the images, returns an empty
  // set of matches.
  KeypointPairing keypoint_matches;
  if (keypoints_and_descriptors_1.descriptors.empty() ||
      keypoints_and_descriptors_2.descriptors.empty()) {
    return keypoint_matches;
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

  // Filter out the keypoint matches to only keep the best ones based on
  // feature-space distance thresholding.
  const int num_matches = feature_matches.size();
  double smallest_feature_distance = 0.0;
  if (num_matches > 0) {
    smallest_feature_distance = feature_matches[0].distance;
    for (int i = 1; i < num_matches; ++i) {
      const double match_distance = feature_matches[i].distance;
      smallest_feature_distance =
          std::min(smallest_feature_distance, match_distance);
    }
  }
  std::vector<cv::DMatch> good_feature_matches;
  const double distance_threshold = std::max(
      kFlannDistanceScalingFactor * smallest_feature_distance,
      kFlannDistanceThreshold);
  for(const cv::DMatch& match : feature_matches) {
    if (match.distance <= distance_threshold) {
      good_feature_matches.push_back(match);
    }
  }

  // Build a list of keypoint match pairs.
  for (const cv::DMatch& match : good_feature_matches) {
    cv::Point2f pixel_loc_1 =
        keypoints_and_descriptors_1.keypoints[match.queryIdx].pt;
    cv::Point2f pixel_loc_2 =
        keypoints_and_descriptors_2.keypoints[match.trainIdx].pt;
    keypoint_matches.first.push_back(pixel_loc_1);
    keypoint_matches.second.push_back(pixel_loc_2);
  }

  return keypoint_matches;
}

}  // namespace

MotionShiftSequence TranslationalRegistration(
    const std::vector<ImageData>& images) {

  // If no images, return an empty sequence.
  if (images.empty()) {
    LOG(WARNING) << "No images given. Returning an empty motion sequence.";
    return MotionShiftSequence();
  }

  // The first image is relative to itself, so its shift is always (0, 0).
  std::vector<MotionShift> motion_shifts;
  motion_shifts.push_back(MotionShift(0, 0));

  // Run keypoint matching between the first image and all other images.
  const KeypointsAndDescriptors& image_0_keypoints =
      DetectKeypoints(images[0]);
  const int num_images = images.size();
  for (int i = 1; i < num_images; ++i) {
    const KeypointsAndDescriptors& image_i_keypoints =
        DetectKeypoints(images[i]);
    const KeypointPairing& keypoint_matches =
        FindMatchingFeatures(image_0_keypoints, image_i_keypoints);
    const KeypointPairing& good_matches = ApplyRANSAC(keypoint_matches);
    //const cv::Mat homography = ComputeHomography(keypoint_matches);
    //LOG(INFO) << "\n" << homography;
    //cv::Mat intrinsics = cv::Mat::ones(cv::Size(3, 3), CV_32F);
    //std::vector<cv::Mat> ignored_rotations, ignored_normals;
    //std::vector<cv::Mat> translations;
    //cv::decomposeHomographyMat(
    //    homography,
    //    intrinsics,
    //    ignored_rotations,
    //    translations,
    //    ignored_normals);
    //LOG(INFO) << "-----";
    //for (const cv::Mat t : translations) {
    //  const double x = t.at<float>(0) / t.at<float>(2);
    //  const double y = t.at<float>(1) / t.at<float>(2);
    //  LOG(INFO) << "T: " << x << ", " << y;
    //}
    //LOG(INFO) << "\n\n";

    //const cv::Mat affine_transform = cv::estimateRigidTransform(
    //    keypoint_matches.first, keypoint_matches.second, true);
    const cv::Mat affine_transform = cv::estimateRigidTransform(
        good_matches.first, good_matches.second, true);
    CHECK(!affine_transform.empty())
        << "Could not determine motion shift between images.";
    const double dx = affine_transform.at<double>(0, 2);
    const double dy = affine_transform.at<double>(1, 2);
    LOG(INFO) << "\n" << affine_transform;
    LOG(INFO) << "T(x, y) = " << dx << ", " << dy;
    motion_shifts.push_back(MotionShift(dx, dy));

//    // TODO: Remove! Replace this with the motion computation (homography).
//    cv::Mat vis;
//    cv::hconcat(
//        images[0].GetVisualizationImage(),
//        images[i].GetVisualizationImage(),
//        vis);
//    const cv::Scalar vis_line_color(0, 255, 0);
//    for (int i = 0; i < good_matches.first.size(); ++i) {
//      cv::line(
//        vis,
//        good_matches.first[i],
//        good_matches.second[i] + cv::Point2f(images[i].GetImageSize().width, 0),
//        vis_line_color,
//        1,  // Line thickness.
//        CV_AA);
//    }
//    cv::imshow("Keypoint Matches", vis);
//    cv::waitKey(0);
//    // ----
  }
  return MotionShiftSequence(motion_shifts);
}

}  // namespace registration
}  // namespace super_resolution
