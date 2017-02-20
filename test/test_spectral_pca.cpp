#include <vector>

#include "hyperspectral/spectral_pca.h"
#include "image/image_data.h"
#include "util/test_util.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::test::AreMatricesEqual;

constexpr double kReconstructionErrorTolerance = 0.00001;

TEST(SpectralPca, Decomposition) {
  // Run on a small controlled data test.
  const cv::Mat small_channel_1 = (cv::Mat_<double>(10, 1)
      << 7, 4, 6, 8, 8, 7, 5, 9, 7, 8);
  const cv::Mat small_channel_2 = (cv::Mat_<double>(10, 1)
      << 4, 1, 3, 6, 5, 2, 3, 5, 4, 2);
  const cv::Mat small_channel_3 = (cv::Mat_<double>(10, 1)
      << 3, 8, 5, 1, 7, 9, 3, 8, 5, 2);
  super_resolution::ImageData small_image;
  small_image.AddChannel(small_channel_1, false);
  small_image.AddChannel(small_channel_2, false);
  small_image.AddChannel(small_channel_3, false);

  // Set up the PCA decomposition.
  const std::vector<super_resolution::ImageData> small_images({
    small_image
  });
  const super_resolution::SpectralPca spectral_pca_small({small_image});

  // Convert the image into PCA space.
  const super_resolution::ImageData small_pca_image =
      spectral_pca_small.GetPcaImage(small_image);

  // Convert it back and make sure the reconstruction matches.
  const super_resolution::ImageData small_image_reconstructed =
      spectral_pca_small.ReconstructImage(small_pca_image);
  for (int channel_index = 0; channel_index < 3; ++channel_index) {
    EXPECT_TRUE(AreMatricesEqual(
        small_image_reconstructed.GetChannelImage(channel_index),
        small_image.GetChannelImage(channel_index),
        kReconstructionErrorTolerance));
  }

  // Run with a bigger image that has controlled strong correlations.
  super_resolution::ImageData hyperspectral_image;
  const int num_channels = 10;
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    const double scalar =
        static_cast<double>(channel_index) / static_cast<double>(num_channels);
    // TODO: add random noise here.
    cv::Mat channel = (cv::Mat_<double>(3, 4)
        << 0.25, 0.5,  0.75, 0.4,
           0.33, 0.66, 0.99, 0.28,
           0.0,  0.5,  1.0,  0.55);
    channel = channel * scalar;
    hyperspectral_image.AddChannel(channel);
  }

  // Test PCA decomposition with no approximations. The reconstruction should
  // be (almost) exact.
  const super_resolution::SpectralPca spectral_pca_full({hyperspectral_image});
  const super_resolution::ImageData hyperspectral_pca_image_full =
      spectral_pca_full.GetPcaImage(hyperspectral_image);
  const super_resolution::ImageData reconstructed_hyperspectral_image_full =
      spectral_pca_full.ReconstructImage(hyperspectral_pca_image_full);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    EXPECT_TRUE(AreMatricesEqual(
        reconstructed_hyperspectral_image_full.GetChannelImage(channel_index),
        hyperspectral_image.GetChannelImage(channel_index),
        kReconstructionErrorTolerance));
  }
}
