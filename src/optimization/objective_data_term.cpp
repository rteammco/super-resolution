#include "optimization/objective_data_term.h"

#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

double ComputeTermForObservation(
    const ImageData& observation,
    const int image_index,
    const ImageModel& image_model,
    const int num_channels,
    const cv::Size& image_size,
    const double* estimated_image_data,
    double* gradient) {

  // Degrade (and re-upsample) the HR estimate with the image model.
  ImageData degraded_hr_image(estimated_image_data, image_size, num_channels);
  image_model.ApplyToImage(&degraded_hr_image, image_index);
  degraded_hr_image.ResizeImage(image_size, INTERPOLATE_NEAREST);

  // Compute the individual residuals by comparing pixel values. Sum them up
  // for the final residual sum.
  double residual_sum = 0;
  const int num_pixels = image_size.width * image_size.height;
  const int num_data_points = num_pixels * num_channels;
  std::vector<double> residuals;
  residuals.reserve(num_data_points);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    const double* degraded_hr_channel_data =
        degraded_hr_image.GetChannelData(channel_index);
    const double* observation_channel_data =
        observation.GetChannelData(channel_index);
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
      const double residual =
          degraded_hr_channel_data[pixel_index] -
          observation_channel_data[pixel_index];
      residuals.push_back(residual);
      residual_sum += (residual * residual);
    }
  }

  // If gradient is not null, apply transpose operations to the residual image.
  // This is used to compute the gradient.
  if (gradient != nullptr) {
    ImageData residual_image(residuals.data(), image_size, num_channels);
    const int scale = image_model.GetDownsamplingScale();
    residual_image.ResizeImage(
        cv::Size(image_size.width / scale, image_size.height / scale),
        INTERPOLATE_NEAREST);
    image_model.ApplyTransposeToImage(&residual_image, image_index);

    // Add to the gradient.
    for (int channel = 0; channel < num_channels; ++channel) {
      const int channel_index = channel * num_pixels;
      const double* residual_channel_data =
          residual_image.GetChannelData(channel);
      for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        const int index = channel_index + pixel_index;
        gradient[index] += 2 * residual_channel_data[pixel_index];
      }
    }
  }

  return residual_sum;
}

double ObjectiveDataTerm::Compute(
    const double* estimated_image_data, double* gradient) const {

  CHECK_NOTNULL(estimated_image_data);

  double residual_sum = 0.0;
  for (int image_index = 0; image_index < observations_.size(); ++image_index) {
    residual_sum += ComputeTermForObservation(
        observations_[image_index],
        image_index,
        image_model_,
        num_channels_,
        image_size_,
        estimated_image_data,
        gradient);
  }
  return residual_sum;
}

}  // namespace super_resolution
