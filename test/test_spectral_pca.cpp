#include <vector>

#include "hyperspectral/spectral_pca.h"
#include "image/image_data.h"
#include "util/matrix_util.h"
#include "util/test_util.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::ImageData;
using super_resolution::test::AreImagesEqual;
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
  ImageData small_image;
  small_image.AddChannel(
      small_channel_1, super_resolution::DO_NOT_NORMALIZE_IMAGE);
  small_image.AddChannel(
      small_channel_2, super_resolution::DO_NOT_NORMALIZE_IMAGE);

  // Set up the PCA decomposition.
  const std::vector<ImageData> small_images({
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
  const ImageData small_pca_image =
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
  const ImageData small_image_reconstructed =
      spectral_pca_small.ReconstructImage(small_pca_image);
  EXPECT_TRUE(AreImagesEqual(
      small_image_reconstructed, small_image, kReconstructionErrorTolerance));

  /* Run with a bigger image that has controlled strong correlations. */

  ImageData hyperspectral_image;
  const int num_channels = 300;
  const cv::Size test_image_size(50, 25);
  const double random_mean = 0.5;
  const double random_stddev = 0.1;
  for (int i = 0; i < num_channels; ++i) {
    // Create a random matrix but compressed similar values in all axes.
    const double scalar =
        static_cast<double>(i) / static_cast<double>(num_channels);
    cv::Mat channel = cv::Mat::ones(
        test_image_size, super_resolution::util::kOpenCvMatrixType);
    cv::randn(channel, cv::Scalar(random_mean), cv::Scalar(random_stddev));
    channel = channel * scalar;
    hyperspectral_image.AddChannel(
        channel, super_resolution::DO_NOT_NORMALIZE_IMAGE);
  }

  // Test PCA decomposition with no approximations. The reconstruction should
  // be (almost) exact.
  const super_resolution::SpectralPca spectral_pca_full({hyperspectral_image});
  const ImageData hyperspectral_pca_image_full =
      spectral_pca_full.GetPcaImage(hyperspectral_image);
  const ImageData reconstructed_hyperspectral_image_full =
      spectral_pca_full.ReconstructImage(hyperspectral_pca_image_full);
  EXPECT_TRUE(AreImagesEqual(
      reconstructed_hyperspectral_image_full,
      hyperspectral_image,
      kReconstructionErrorTolerance));

  // Test PCA decomposition using only a subset of bands and check that the
  // reconstruction is a close-enough approximation.
  const super_resolution::SpectralPca spectral_pca_approx_count(
      {hyperspectral_image}, 250);
  const ImageData hyperspectral_pca_image_approx_count =
      spectral_pca_approx_count.GetPcaImage(hyperspectral_image);
  const ImageData reconstructed_hyperspectral_image_approx_count =
      spectral_pca_approx_count.ReconstructImage(
          hyperspectral_pca_image_approx_count);
  EXPECT_TRUE(AreImagesEqual(
      reconstructed_hyperspectral_image_approx_count,
      hyperspectral_image,
      0.05));

  // Finally test using a required retained variance amount (99%). This should
  // limit the number of PCA bands but still yield a close approximation in the
  // reconstruction.
  const super_resolution::SpectralPca spectral_pca_approx_var(
      {hyperspectral_image}, 0.999);
  const ImageData hyperspectral_pca_image_approx_var =
      spectral_pca_approx_var.GetPcaImage(hyperspectral_image);
  EXPECT_LT(
      hyperspectral_pca_image_approx_var.GetNumChannels(),
      hyperspectral_image.GetNumChannels());
  const ImageData reconstructed_hyperspectral_image_approx_var =
      spectral_pca_approx_var.ReconstructImage(
          hyperspectral_pca_image_approx_var);
  EXPECT_TRUE(AreImagesEqual(
      reconstructed_hyperspectral_image_approx_var,
      hyperspectral_image,
      0.05));
}
