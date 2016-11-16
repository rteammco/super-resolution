#include "solvers/map_cost_processor.h"

#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {

MapCostProcessor::MapCostProcessor(
    const std::vector<ImageData>& low_res_images,
    const ImageModel& image_model,
    const cv::Size& image_size)
    : image_model_(image_model), image_size_(image_size) {

  for (const ImageData& low_res_image : low_res_images) {
    ImageData observation = low_res_image;  // copy
    observation.ResizeImage(image_size_, cv::INTER_AREA);
    observations_.push_back(observation);
  }
}

std::vector<double> MapCostProcessor::ComputeDataTermResiduals(
    const int image_index,
    const int channel_index,
    const double* estimated_image_data) const {

  // Convert the pixel data into an OpenCV image.
  const cv::Mat hr_image_estimate(
      image_size_,
      util::kOpenCvMatrixType,
      const_cast<void*>(reinterpret_cast<const void*>(estimated_image_data)));

  // Degrade (and re-upsample) the HR estimate with the image model.
  // TODO: this currently assumes there is only one channel in the image.
  // TODO: would be nice if we can build ImageData directly from an array,
  //       skipping the above step.
  const ImageData hr_image_data(hr_image_estimate);
  ImageData degraded_hr_image =
      image_model_.ApplyModel(hr_image_data, image_index);
  degraded_hr_image.ResizeImage(image_size_, cv::INTER_AREA);

  // Compute the residuals by comparing.
  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    const double difference =
        degraded_hr_image.GetPixelValue(0, i) -  // TODO: channel 0 hardcoded
        observations_.at(image_index).GetPixelValue(channel_index, i);
    LOG(INFO) << "channel " << channel_index
              << ", pixel " << i << " = " << difference; // TODO: remove
    residuals.push_back(difference);
  }
  return residuals;
}

std::vector<double> MapCostProcessor::ComputeRegularizationResiduals(
    const int channel_index,
    const double* estimated_image_data) const {

  // TODO: implement
  const int num_pixels = image_size_.width * image_size_.height;
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    residuals.push_back(0);
  }
  return residuals;
}

}  // namespace super_resolution
