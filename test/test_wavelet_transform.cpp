#include <string>

#include "image/image_data.h"
#include "util/data_loader.h"
#include "util/util.h"
#include "wavelet/wavelet_transform.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

static const std::string kTestImagePath = "../test_data/dallas.jpg";

TEST(WaveletTransform, WaveletTransform) {
  const super_resolution::ImageData original_image =
      super_resolution::util::LoadImage(kTestImagePath);
  super_resolution::wavelet::WaveletCoefficients coefficients =
      super_resolution::wavelet::WaveletTransform(original_image);

  // TODO: Do an actual test, not display.
  super_resolution::util::DisplayImagesSideBySide({
      original_image,
      coefficients.ll,
      coefficients.lh,
      coefficients.hl,
      coefficients.hh
  });
}
