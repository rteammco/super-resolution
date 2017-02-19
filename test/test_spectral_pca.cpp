#include <vector>

#include "hyperspectral/spectral_pca.h"
#include "image/image_data.h"

#include "opencv2/core/core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(SpectralPca, Decomposition) {
  const cv::Mat channel = (cv::Mat_<double>(3, 3)
      << 0.25, 0.5, 0.75,
         0.33, 0.66, 0.99,
         0.0, 0.5, 1.0);
  super_resolution::ImageData image;
  image.AddChannel(channel);
  image.AddChannel(channel);
  image.AddChannel(channel);
  image.AddChannel(channel);
  const std::vector<super_resolution::ImageData> images({
    image
  });
  const super_resolution::SpectralPca spectral_pca(images);

  // TODO: test GetPcaImage()!
  // TODO: test ReconstructImage()!
}
