#include "wavelet/wavelet_transform.h"

#include "opencv2/core/core.hpp"

#include "image/image_data.h"

namespace super_resolution {
namespace wavelet {

WaveletCoefficients WaveletTransform(const ImageData& image) {
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
  // TODO: implement.
  return coefficients.ll;
}

}  // namespace wavelet
}  // namespace super_resolution
