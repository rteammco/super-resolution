#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::DownsamplingModule;
using super_resolution::ImageModel;

// Returns true if the two given matrices contain identical values.
// Source:
//   http://stackoverflow.com/questions/9905093/how-to-check-whether-two-matrixes-are-identical-in-opencv  NOLINT
bool AreMatricesEqual(const cv::Mat& mat1, const cv::Mat& mat2) {
  if (mat1.empty() && mat2.empty()) {
    return true;
  }
  if (mat1.cols != mat2.cols ||
      mat1.rows != mat2.rows ||
      mat1.dims != mat2.dims) {
    return false;
  }

  cv::Mat diff;
  cv::compare(mat1, mat2, diff, cv::CMP_NE);
  return cv::countNonZero(diff) == 0;
}

TEST(ImageModel, AdditiveNoiseModule) {
  // TODO: implement
}

TEST(ImageModel, DownsamplingModule) {
  const int downsampling_scale = 2;
  const int num_high_res_pixels = 24;
  const int num_low_res_pixels = 6;  // 24 / (2*2)

  const cv::Mat expected_matrix = (cv::Mat_<double>(
      num_low_res_pixels, num_high_res_pixels) <<
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);

  DownsamplingModule downsampling_module(downsampling_scale);
  const cv::Mat downsampling_matrix =
      downsampling_module.GetOperatorMatrix(num_high_res_pixels, 0);

  EXPECT_TRUE(AreMatricesEqual(downsampling_matrix, expected_matrix));
}

TEST(ImageModel, MotionModule) {
  // TODO: implement
}

TEST(ImageModel, PsfBlurModule) {
  // TODO: implement
}

TEST(ImageModel, TestGetOperatorMatrix) {
//  [[1, 2, 3, 4, 5, 6],
//   [7, 8, 9, 0, 1, 2],
//   [9, 7, 5, 4, 2, 1],
//   [2, 4, 6, 8, 0, 1]])
  // TODO: implement
}
