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
  // Run on a small controlled data test. Example and known truths are from:
  //   https://www.youtube.com/watch?v=VzPpJXISz-E
  const cv::Mat small_channel_1 = (cv::Mat_<double>(10, 1)
      << 1.85, 2.05, -0.95, -1.55, -2.55, 2.85, 1.95, 2.75, -2.75, -3.65);
  const cv::Mat small_channel_2 = (cv::Mat_<double>(10, 1)
      << 2.2175, 2.5425, -1.2075, -1.9575, -3.3825,
         3.6425, 2.5925, 3.3175, -3.4825, -4.2825);
  super_resolution::ImageData small_image;
  small_image.AddChannel(small_channel_1, false);
  small_image.AddChannel(small_channel_2, false);

  // Set up the PCA decomposition.
  const std::vector<super_resolution::ImageData> small_images({
    small_image
  });
  const super_resolution::SpectralPca spectral_pca_small({small_image});

  // Convert the image into PCA space and compare to known ground truth.
  const cv::Mat small_pca_known_channel_1 = (cv::Mat_<double>(10, 1)
      << 2.88737, 3.266, -1.53633, -2.49680, -4.23402,
         4.62459, 3.24237, 4.30858, -4.43722, -5.62453);
  const cv::Mat small_pca_known_channel_2 = (cv::Mat_<double>(10, 1)
      << 0.0538, 0.00622, 0.01545, 0.01729, 0.12995,
         -0.05886, -0.10306, 0.06669, 0.03664, -0.16411);
  const super_resolution::ImageData small_pca_image =
      spectral_pca_small.GetPcaImage(small_image);
  EXPECT_EQ(small_pca_image.GetNumChannels(), 2);
  EXPECT_TRUE(AreMatricesEqual(
      small_pca_image.GetChannelImage(0),
      small_pca_known_channel_1,
      kReconstructionErrorTolerance));
  EXPECT_TRUE(AreMatricesEqual(
      small_pca_image.GetChannelImage(1),
      small_pca_known_channel_2,
      kReconstructionErrorTolerance));

  // Convert it back and make sure the reconstruction matches.
  const super_resolution::ImageData small_image_reconstructed =
      spectral_pca_small.ReconstructImage(small_pca_image);
  for (int i = 0; i < small_image.GetNumChannels(); ++i) {
    EXPECT_TRUE(AreMatricesEqual(
        small_image_reconstructed.GetChannelImage(i),
        small_image.GetChannelImage(i),
        kReconstructionErrorTolerance));
  }

  /* Run with a bigger image that has controlled strong correlations. */

  // TODO: many more pixels and more channels.
  super_resolution::ImageData hyperspectral_image;
  const int num_channels = 10;
  for (int i = 0; i < num_channels; ++i) {
    const double scalar =
        static_cast<double>(i) / static_cast<double>(num_channels);
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
  for (int i = 0; i < num_channels; ++i) {
    EXPECT_TRUE(AreMatricesEqual(
        reconstructed_hyperspectral_image_full.GetChannelImage(i),
        hyperspectral_image.GetChannelImage(i),
        kReconstructionErrorTolerance));
  }

  // TODO: test with partial reconstruction.
}
