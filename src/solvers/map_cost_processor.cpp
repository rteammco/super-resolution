#include "solvers/map_cost_processor.h"

#include <vector>

#include "image/image_data.h"
#include "image_model/image_model.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

MapCostProcessor::MapCostProcessor(
    const std::vector<ImageData>& low_res_images,
    const ImageModel& image_model,
    const cv::Size& high_res_image_size)
    : image_model_(image_model), high_res_image_size_(high_res_image_size) {

  for (const ImageData& low_res_image : low_res_images) {
    ImageData observation = low_res_image;  // copy
    observation.ResizeImage(high_res_image_size_, cv::INTER_AREA);
    observations_.push_back(observation);
  }
}

std::vector<double> MapCostProcessor::ComputeDataTermResiduals(
    const int image_index,
    const int channel_index,
    const std::vector<double>& high_res_image_data) const {

  // TODO: implement
  const int num_pixels = high_res_image_data.size();
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    residuals.push_back(0);
  }
  return residuals;
}

std::vector<double> MapCostProcessor::ComputeRegularizationResiduals(
    const int channel_index,
    const std::vector<double>& high_res_image_data) const {

  // TODO: implement
  const int num_pixels = high_res_image_data.size();
  std::vector<double> residuals;
  residuals.reserve(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    residuals.push_back(0);
  }
  return residuals;
}

}  // namespace super_resolution
