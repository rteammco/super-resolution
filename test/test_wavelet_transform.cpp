#include <string>

#include "image/image_data.h"
#include "util/data_loader.h"
#include "util/test_util.h"
#include "util/util.h"
#include "wavelet/wavelet_transform.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::util::GetAbsoluteCodePath;

static const std::string kTestImagePath =
    GetAbsoluteCodePath("test_data/dallas.jpg");
constexpr double kReconstructionErrorTolerance = 1.0 / 255.0;

TEST(WaveletTransform, WaveletTransform) {
  const super_resolution::ImageData original_image =
      super_resolution::util::LoadImage(kTestImagePath);

  // Wavelet transform (DWT).
  super_resolution::wavelet::WaveletCoefficients coefficients =
      super_resolution::wavelet::WaveletTransform(original_image);

  // Reconstruct the image (iDWT).
  const super_resolution::ImageData reconstructed_image =
      super_resolution::wavelet::InverseWaveletTransform(coefficients);

  EXPECT_EQ(
      reconstructed_image.GetNumChannels(), original_image.GetNumChannels());
  EXPECT_EQ(reconstructed_image.GetImageSize(), original_image.GetImageSize());
  EXPECT_TRUE(super_resolution::test::AreImagesEqual(
      original_image, reconstructed_image, kReconstructionErrorTolerance));

  // TODO: Remove this display code.
  // super_resolution::util::DisplayImagesSideBySide({
  //     original_image,
  //     reconstructed_image,
  //     coefficients.GetCoefficientsImage()
  // });
}
