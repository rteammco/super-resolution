#include "image/image_data.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::ImageData;

using testing::DoubleEq;

constexpr double kPixelErrorTolerance = 1.0 / 255.0;

// This test verifies that channels are added correctly to ImageData and pixels
// and channels in the image can be accessed and manipulated correctly.
TEST(ImageData, AddAndAccessImageData) {
  const int num_test_rows = 3;
  const int num_test_cols = 5;

  ImageData image_data;

  /* Verify behavior of a an empty image. */

  EXPECT_EQ(image_data.GetNumChannels(), 0);
  EXPECT_EQ(image_data.GetImageSize(), cv::Size(0, 0));
  EXPECT_EQ(image_data.GetNumPixels(), 0);

  /* Verify behavior of a single-channel image. */

  // Convert to CV_8UC1 (unsigned char) 0-255 range image first.
  cv::Mat channel_0 = (cv::Mat_<double>(num_test_rows, num_test_cols)
      << 0.1,  0.2,  0.3,  0.4,  0.5,
         0.15, 0.25, 0.35, 0.45, 0.55,
         0.6,  0.65, 0.7,  0.75, 0.8);
  cv::Mat channel_0_converted;
  channel_0.convertTo(channel_0_converted, CV_8UC1, 255);
  image_data.AddChannel(channel_0_converted);

  EXPECT_EQ(image_data.GetNumChannels(), 1);
  EXPECT_EQ(image_data.GetImageSize(), cv::Size(num_test_cols, num_test_rows));
  EXPECT_EQ(image_data.GetNumPixels(), num_test_rows * num_test_cols);

  // Check pixel access values.
  EXPECT_NEAR(image_data.GetPixelValue(0, 0), 0.1, kPixelErrorTolerance);
  EXPECT_NEAR(image_data.GetPixelValue(0, 2), 0.3, kPixelErrorTolerance);
  EXPECT_NEAR(image_data.GetPixelValue(0, 8), 0.45, kPixelErrorTolerance);
  EXPECT_NEAR(image_data.GetPixelValue(0, 11), 0.65, kPixelErrorTolerance);

  // Check that the returned channel image matches.
  cv::Mat returned_channel_0 = image_data.GetChannelImage(0);
  for (int row = 0; row < num_test_rows; ++row) {
    for (int col = 0; col < num_test_cols; ++col) {
      EXPECT_NEAR(
          returned_channel_0.at<double>(row, col),
          channel_0.at<double>(row, col),
          kPixelErrorTolerance);
    }
  }

  // Check data pointer access.
  double* pixel_ptr = image_data.GetMutableDataPointer(0, 0);
  EXPECT_NEAR(pixel_ptr[0], 0.1, kPixelErrorTolerance);
  pixel_ptr = image_data.GetMutableDataPointer(0, 3);
  EXPECT_NEAR(pixel_ptr[0], 0.4, kPixelErrorTolerance);
  pixel_ptr = image_data.GetMutableDataPointer(0, 4);
  EXPECT_NEAR(pixel_ptr[0], 0.5, kPixelErrorTolerance);
  pixel_ptr = image_data.GetMutableDataPointer(0, 14);
  EXPECT_NEAR(pixel_ptr[0], 0.8, kPixelErrorTolerance);

  // Check data manipulation through the pointers works as expected.
  // Change all pixel values to 0.33 and expect the image to be updated.
  const double new_pixel_value = 0.33;
  for (int i = 0; i < num_test_rows * num_test_cols; ++i) {
    pixel_ptr = image_data.GetMutableDataPointer(0, i);
    pixel_ptr[0] = new_pixel_value;
  }
  // Check that all returned pixel values are updated.
  for (int i = 0; i < num_test_rows * num_test_cols; ++i) {
    EXPECT_NEAR(
        image_data.GetPixelValue(0, i), new_pixel_value, kPixelErrorTolerance);
  }
  // Check that the returned channel image has also been updated.
  returned_channel_0 = image_data.GetChannelImage(0);
  for (int row = 0; row < num_test_rows; ++row) {
    for (int col = 0; col < num_test_cols; ++col) {
      EXPECT_NEAR(
          returned_channel_0.at<double>(row, col),
          new_pixel_value,
          kPixelErrorTolerance);
    }
  }

  // TODO: check that the channel got inserted as a copy and that the original
  // image was not actually modified.

  /* Verify behavior with multiple channels. */

  // Add 10 more channels.
  for (int i = 0; i < 10; ++i) {
    const double pixel_value = 1.0 / static_cast<double>(i+1);
    cv::Mat next_channel(num_test_rows, num_test_cols, CV_64FC1);
    next_channel = cv::Scalar(pixel_value);
    cv::Mat next_channel_converted;
    next_channel.convertTo(next_channel_converted, CV_8UC1, 255);
    image_data.AddChannel(next_channel_converted);
  }

  EXPECT_EQ(image_data.GetNumChannels(), 11);
  EXPECT_EQ(image_data.GetImageSize(), cv::Size(num_test_cols, num_test_rows));
  EXPECT_EQ(image_data.GetNumPixels(), num_test_rows * num_test_cols);

  // TODO: check that we can access pixels in each channel.
  // TODO: check that we can access the data pointer in each channel.
  // TODO: check that we can manipulate the data pointer in each channel.

  // Verify image type is consistent.
  EXPECT_EQ(image_data.GetOpenCvImageType(), CV_64FC1);
}

// This test verifies that the copy constructor works as expected.
TEST(ImageData, CopyConstructor) {
  // TODO: implement.
  // ImageData(const ImageData& other)
}

// This test verifies that the constructor which takes an OpenCV image as input
// works as expected, and correctly splits up the channels.
TEST(ImageData, FromOpenCvImageConstructor) {
  // TODO: implement.
  // ImageData(const cv::Mat& image)
}

// This test verifies that the image is correctly resized, with one or more
// channels.
TEST(ImageData, ResizeImage) {
  // TODO: implement.
  // ResizeImage(scale, interp.meth)
}

// This test verifies that the correct visualization image is returned for
// different numbers of channels.
TEST(ImageData, GetVisualizationImage) {
  // TODO: implement.
  // cv::Mat GetVisualizationImage
}
