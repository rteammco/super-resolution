#include "wavelet/wavelet_transform.h"

#include "opencv2/core/core.hpp"

#include "image/image_data.h"

#include "glog/logging.h"

namespace super_resolution {
namespace wavelet {

ImageData WaveletCoefficients::GetCoefficientsImage() const {
  const int num_channels = ll.GetNumChannels();
  CHECK_GT(num_channels, 0)
      << "Coefficient images may not be empty.";

  CHECK_EQ(num_channels, lh.GetNumChannels())
      << "All coefficients must have the same number of channels.";
  CHECK_EQ(num_channels, hl.GetNumChannels())
      << "All coefficients must have the same number of channels.";
  CHECK_EQ(num_channels, hh.GetNumChannels())
      << "All coefficients must have the same number of channels.";

  const cv::Size coefficients_size = ll.GetImageSize();
  CHECK_EQ(coefficients_size, lh.GetImageSize())
      << "All coefficients must be the same size.";
  CHECK_EQ(coefficients_size, hl.GetImageSize())
      << "All coefficients must be the same size.";
  CHECK_EQ(coefficients_size, hh.GetImageSize())
      << "All coefficients must be the same size.";

  const int width = coefficients_size.width;
  const int height = coefficients_size.height;
  const cv::Size visualization_image_size(width * 2, height * 2);
  ImageData stitched_image;
  for (int channel = 0; channel < num_channels; ++channel) {
    const cv::Mat channel_ll = ll.GetChannelImage(channel);
    const cv::Mat channel_lh = lh.GetChannelImage(channel);
    const cv::Mat channel_hl = hl.GetChannelImage(channel);
    const cv::Mat channel_hh = hh.GetChannelImage(channel);
    cv::Mat channel_image(visualization_image_size, channel_ll.type());

    cv::Mat top_left =
        channel_image(cv::Rect(0, 0, width, height));
    channel_ll.copyTo(top_left);

    cv::Mat top_right = channel_image(cv::Rect(width, 0, width, height));
    channel_lh.copyTo(top_right);

    cv::Mat bottom_left = channel_image(cv::Rect(0, height, width, height));
    channel_hl.copyTo(bottom_left);

    cv::Mat bottom_right =
        channel_image(cv::Rect(width, height, width, height));
    channel_hh.copyTo(bottom_right);

    stitched_image.AddChannel(channel_image, DO_NOT_NORMALIZE_IMAGE);
  }

  return stitched_image;
}

WaveletCoefficients WaveletTransform(const ImageData& image) {
  CHECK_GT(image.GetNumChannels(), 0) << "Image cannot be empty.";

  const cv::Size image_size = image.GetImageSize();
  const cv::Size target_size(image_size.width / 2, image_size.height / 2);
  WaveletCoefficients coefficients;

  // TODO: This is haar filter only.
  // TODO: Can this be done with a convolution?
  for (int channel = 0; channel < image.GetNumChannels(); ++channel) {
    const cv::Mat channel_image = image.GetChannelImage(channel);
    cv::Mat channel_ll(target_size, channel_image.type());
    cv::Mat channel_lh(target_size, channel_image.type());
    cv::Mat channel_hl(target_size, channel_image.type());
    cv::Mat channel_hh(target_size, channel_image.type());

    // This code has been adapted from
    // http://stackoverflow.com/questions/20071854/wavelet-transform-in-opencv
    for (int row = 0; row < target_size.height; ++row) {
      for (int col = 0; col < target_size.width; ++col) {
        channel_ll.at<double>(row, col) = 0.5 * (
            channel_image.at<double>(2 * row, 2 * col)
            + channel_image.at<double>(2 * row, 2 * col + 1)
            + channel_image.at<double>(2 * row + 1, 2 * col)
            + channel_image.at<double>(2 * row + 1, 2 * col + 1));

        channel_lh.at<double>(row, col) = 0.5 * (
            channel_image.at<double>(2 * row, 2 * col)
            - channel_image.at<double>(2 * row, 2 * col + 1)
            + channel_image.at<double>(2 * row + 1, 2 * col)
            - channel_image.at<double>(2 * row + 1, 2 * col + 1));

        channel_hl.at<double>(row, col) = 0.5 * (
            channel_image.at<double>(2 * row, 2 * col)
            + channel_image.at<double>(2 * row, 2 * col + 1)
            - channel_image.at<double>(2 * row + 1, 2 * col)
            - channel_image.at<double>(2 * row + 1, 2 * col + 1));

        channel_hh.at<double>(row, col) = 0.5 * (
            channel_image.at<double>(2 * row, 2 * col)
            - channel_image.at<double>(2 * row, 2 * col + 1)
            - channel_image.at<double>(2 * row + 1, 2 * col)
            + channel_image.at<double>(2 * row + 1, 2 * col + 1));
      }
    }
    coefficients.ll.AddChannel(channel_ll, DO_NOT_NORMALIZE_IMAGE);
    coefficients.lh.AddChannel(channel_lh, DO_NOT_NORMALIZE_IMAGE);
    coefficients.hl.AddChannel(channel_hl, DO_NOT_NORMALIZE_IMAGE);
    coefficients.hh.AddChannel(channel_hh, DO_NOT_NORMALIZE_IMAGE);
  }

  return coefficients;
}

ImageData InverseWaveletTransform(const WaveletCoefficients& coefficients) {
  const int num_channels = coefficients.ll.GetNumChannels();
  CHECK_GT(num_channels, 0)
      << "Coefficient images may not be empty.";

  CHECK_EQ(num_channels, coefficients.lh.GetNumChannels())
      << "All coefficients must have the same number of channels.";
  CHECK_EQ(num_channels, coefficients.hl.GetNumChannels())
      << "All coefficients must have the same number of channels.";
  CHECK_EQ(num_channels, coefficients.hh.GetNumChannels())
      << "All coefficients must have the same number of channels.";

  const cv::Size coefficients_size = coefficients.ll.GetImageSize();
  CHECK_EQ(coefficients_size, coefficients.lh.GetImageSize())
      << "All coefficients must be the same size.";
  CHECK_EQ(coefficients_size, coefficients.hl.GetImageSize())
      << "All coefficients must be the same size.";
  CHECK_EQ(coefficients_size, coefficients.hh.GetImageSize())
      << "All coefficients must be the same size.";

  // TODO: Can this be done with more efficiently by utilizing OpenCV?
  ImageData reconstructed_image;
  const cv::Size original_size(
      coefficients_size.width * 2, coefficients_size.height * 2);
  for (int channel = 0; channel < coefficients.ll.GetNumChannels(); ++channel) {
    const cv::Mat channel_ll = coefficients.ll.GetChannelImage(channel);
    const cv::Mat channel_lh = coefficients.lh.GetChannelImage(channel);
    const cv::Mat channel_hl = coefficients.hl.GetChannelImage(channel);
    const cv::Mat channel_hh = coefficients.hh.GetChannelImage(channel);
    cv::Mat channel_image(original_size, channel_ll.type());

    // This code has been adapted from
    // http://stackoverflow.com/questions/20071854/wavelet-transform-in-opencv
    for (int row = 0; row < coefficients_size.height; ++row) {
      for (int col = 0; col < coefficients_size.width; ++col) {
        const double ll_value = channel_ll.at<double>(row, col);
        const double lh_value = channel_lh.at<double>(row, col);
        const double hl_value = channel_hl.at<double>(row, col);
        const double hh_value = channel_hh.at<double>(row, col);

        // TODO: Shrinkage?

        channel_image.at<double>(row * 2, col * 2) =
            0.5 * (ll_value + lh_value + hl_value + hh_value);
        channel_image.at<double>(row * 2, col * 2 + 1) =
            0.5 * (ll_value - lh_value + hl_value - hh_value);
        channel_image.at<double>(row * 2 + 1, col * 2) =
            0.5 * (ll_value + lh_value - hl_value - hh_value);
        channel_image.at<double>(row * 2 + 1, col * 2 + 1) =
            0.5 * (ll_value - lh_value - hl_value + hh_value);
      }
    }
    reconstructed_image.AddChannel(channel_image, DO_NOT_NORMALIZE_IMAGE);
  }

  return reconstructed_image;
}

}  // namespace wavelet
}  // namespace super_resolution
