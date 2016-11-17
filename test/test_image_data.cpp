#include <limits>

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::ImageData;

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
  cv::Mat channel_0_original_clone = channel_0.clone();
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

  // Check that another ImageData created with pre-normalized values (between 0
  // and 1 instead of between 0 and 255) will have identical data.
  ImageData image_data2(channel_0);  // channel_0 is NOT converted.
  const int num_pixels = num_test_rows * num_test_cols;
  for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
    EXPECT_NEAR(
        image_data.GetPixelValue(0, pixel_index),  // channel 0, pixel_index
        image_data2.GetPixelValue(0, pixel_index),
        kPixelErrorTolerance);
  }

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
  double* pixel_ptr = image_data.GetMutableDataPointer(0);
  EXPECT_NEAR(pixel_ptr[0], 0.1, kPixelErrorTolerance);
  EXPECT_NEAR(pixel_ptr[3], 0.4, kPixelErrorTolerance);
  EXPECT_NEAR(pixel_ptr[4], 0.5, kPixelErrorTolerance);
  EXPECT_NEAR(pixel_ptr[14], 0.8, kPixelErrorTolerance);

  // Check data manipulation through the pointers works as expected.
  // Change all pixel values to 0.33 and expect the image to be updated.
  const double new_pixel_value = 0.33;
  for (int i = 0; i < num_test_rows * num_test_cols; ++i) {
    pixel_ptr[i] = new_pixel_value;
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

  // Check that the image channel_0_converted which got inserted as a channel
  // got inserted as a copy and that the original image was not actually
  // modified.
  cv::Mat channel_0_clone_converted;
  channel_0_original_clone.convertTo(channel_0_clone_converted, CV_8UC1, 255);
  for (int row = 0; row < num_test_rows; ++row) {
    for (int col = 0; col < num_test_cols; ++col) {
      EXPECT_EQ(
          channel_0_converted.at<uchar>(row, col),
          channel_0_clone_converted.at<uchar>(row, col));
    }
  }

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
}

// This test verifies that the copy constructor works as expected.
TEST(ImageData, CopyConstructor) {
  // Create an ImageData object with 10 channels.
  ImageData image_data;
  for (int i = 0; i < 10; ++i) {
    cv::Mat next_channel(25, 25, CV_8UC1);
    next_channel = cv::Scalar(5 * i);  // single intensity is 5 * i
    image_data.AddChannel(next_channel);
  }

  // Run some standard checks.
  EXPECT_EQ(image_data.GetNumChannels(), 10);
  EXPECT_EQ(image_data.GetImageSize(), cv::Size(25, 25));
  EXPECT_EQ(image_data.GetNumPixels(), 25 * 25);

  // Copy the ImageData and verify that the new object matches the old object.
  ImageData image_data2 = image_data;
  EXPECT_EQ(image_data2.GetNumChannels(), 10);
  EXPECT_EQ(image_data2.GetImageSize(), cv::Size(25, 25));
  EXPECT_EQ(image_data2.GetNumPixels(), 25 * 25);

  for (int channel_index = 0; channel_index < 10; ++channel_index) {
    for (int pixel_index = 0; pixel_index < 25 * 25; ++pixel_index) {
      EXPECT_NEAR(
          image_data.GetPixelValue(channel_index, pixel_index),
          image_data2.GetPixelValue(channel_index, pixel_index),
          std::numeric_limits<double>::epsilon());
    }
  }

  // Check that the new ImageData is a clone, and changing the data will not
  // affect the old ImageData object.
  // TODO: do this.
}

// This test verifies that the constructor which takes an OpenCV image as input
// works as expected, and correctly splits up the channels.
TEST(ImageData, FromOpenCvImageConstructor) {
  // TODO: implement.
  // ImageData(const cv::Mat& image)

  /* Verify the functionality of the manual normalization constructor. */

  const cv::Mat invalid_image = (cv::Mat_<double>(3, 3)
      << 0.5, 1.5,  100,
         -25, 0.0,  -30,
         55,  1.98, 1000);
  ImageData image_data_not_normalized(invalid_image, false);
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(
        invalid_image.at<double>(i),
        image_data_not_normalized.GetPixelValue(0, i),
        std::numeric_limits<double>::epsilon());
  }
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
