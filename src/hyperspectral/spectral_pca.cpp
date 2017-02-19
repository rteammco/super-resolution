#include "hyperspectral/spectral_pca.h"

#include <algorithm>
#include <vector>

#include "image/image_data.h"
#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

cv::Mat GetPcaInputData(const std::vector<ImageData>& hyperspectral_images) {
  CHECK(!hyperspectral_images.empty())
      << "At least one image is required to compute the PCA basis.";

  // Make sure we have the right number of channels. Also it does not make
  // sense to do this on non-hyperspectral images, so warn the user if that's
  // the case.
  const int num_channels = hyperspectral_images[0].GetNumChannels();
  CHECK_GT(num_channels, 0) << "Cannot compute PCA on empty images.";
  if (num_channels <= 3) {
    LOG(WARNING)
        << "The given images do not appear to be hyperspectral "
        << "(3 or fewer channels). PCA decomposition may not be "
        << "useful or applicable here.";
  }

  // Format the input data as pixel vectors for PCA.
  // TODO: Probably need to do subsampling. Images can be way too big.
  const int num_images = hyperspectral_images.size();
  const int num_pixels = hyperspectral_images[0].GetNumPixels();
  const int num_data_points = num_images * num_pixels;
  cv::Mat input_data(num_data_points, num_channels, util::kOpenCvMatrixType);
  for (int image_index = 0; image_index < num_images; ++image_index) {
    const ImageData& image = hyperspectral_images[image_index];
    CHECK_EQ(image.GetNumChannels(), num_channels)
        << "Inconsistent number of channels between the given images. "
        << "Cannot perform PCA.";
    // TODO: Faster to use cv::ROI and unfold the channel matrices that way.
    for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
      for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        const int data_row = image_index * num_pixels + pixel_index;
        input_data.at<double>(data_row, channel_index) =
            image.GetPixelValue(channel_index, pixel_index);
      }
    }
  }
  return input_data;
}

SpectralPca::SpectralPca(
    const std::vector<ImageData>& hyperspectral_images,
    const int num_pca_bands) {

  const cv::Mat input_data = GetPcaInputData(hyperspectral_images);
  pca_ = cv::PCA(input_data, cv::Mat(), CV_PCA_DATA_AS_ROW, num_pca_bands);

  // Set the number of spectral in the original and PCA spaces.
  const cv::Size eigenvector_matrix_size = pca_.eigenvectors.size();
  num_spectral_bands_ = eigenvector_matrix_size.width;
  num_pca_bands_ = eigenvector_matrix_size.height;
}

SpectralPca::SpectralPca(
    const std::vector<ImageData>& hyperspectral_images,
    const double retained_variance) {

  const cv::Mat input_data = GetPcaInputData(hyperspectral_images);
  pca_ = cv::PCA(input_data, cv::Mat(), CV_PCA_DATA_AS_ROW, retained_variance);

  // Set the number of spectral in the original and PCA spaces.
  const cv::Size eigenvector_matrix_size = pca_.eigenvectors.size();
  num_spectral_bands_ = eigenvector_matrix_size.width;
  num_pca_bands_ = eigenvector_matrix_size.height;
}

ImageData SpectralPca::GetPcaImage(const ImageData& image_data) const {
  CHECK_EQ(image_data.GetNumChannels(), num_spectral_bands_)
      << "The input image does not have the correct number of channels.";

  // Create empty OpenCV images for the PCA image channels.
  std::vector<cv::Mat> pca_image_channels;
  pca_image_channels.reserve(num_pca_bands_);
  for (int i = 0; i < num_pca_bands_; ++i) {
    cv::Mat channel_image =
        cv::Mat::zeros(image_data.GetImageSize(), util::kOpenCvMatrixType);
    pca_image_channels.push_back(channel_image);
  }

  // Project the original image into PCA space pixel by pixel.
  const int num_pixels = image_data.GetNumPixels();
  for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
    // Extract the pixel vector from the original image.
    cv::Mat pixel_vector(num_spectral_bands_, 1, util::kOpenCvMatrixType);
    for (int i = 0; i < num_spectral_bands_; ++i) {
      pixel_vector.at<double>(i) = image_data.GetPixelValue(i, pixel_index);
    }
    // Project the pixel vector into PCA space and set the PCA channel values.
    cv::Mat pca_pixel_vector = pca_.project(pixel_vector);
    for (int i = 0; i< num_pca_bands_; ++i) {
      pca_image_channels[i].at<double>(pixel_index) =
          pca_pixel_vector.at<double>(i);
    }
  }

  // Return the projected image.
  ImageData pca_image;
  for (const cv::Mat& channel_image : pca_image_channels) {
    pca_image.AddChannel(channel_image);
    // TODO: set PCA image spectral mode to PCA (manually).
  }
  return pca_image;
}

ImageData SpectralPca::ReconstructImage(const ImageData& pca_image_data) const {
  // TODO: this process is identical to the process for GetPcaImage(), but the
  // channel counts are opposite and use pca_.backProject() instead of
  // pca_.project().
}

}  // namespace super_resolution
