#include "image_model/image_model.h"

#include <memory>
#include <utility>

#include "image/image_data.h"
#include "image_model/degradation_operator.h"

#include "glog/logging.h"

namespace super_resolution {

void ImageModel::AddDegradationOperator(
    std::unique_ptr<DegradationOperator> degradation_operator) {

  degradation_operators_.push_back(std::move(degradation_operator));
}

ImageData ImageModel::ApplyToImage(
    const ImageData& image_data, const int index) const {

  ImageData degraded_image = image_data;
  for (const auto& degradation_operator : degradation_operators_) {
    degradation_operator->ApplyToImage(&degraded_image, index);
  }
  return degraded_image;
}

void ImageModel::ApplyToImage(ImageData* image_data, const int index) const {
  for (const auto& degradation_operator : degradation_operators_) {
    degradation_operator->ApplyToImage(image_data, index);
  }
}

double ImageModel::ApplyToPixel(
    const ImageData& image_data,
    const int image_index,
    const int channel_index,
    const int pixel_index) const {

  // Sum up the patch radii required by each operator and create a patch of
  // that size cropped out from the given image data.
  int patch_radius = 0;
  for (const auto& degradation_operator : degradation_operators_) {
    patch_radius += degradation_operator->GetPixelPatchRadius();
  }

  // TODO: implement all of this and test it. That's hopefully the solution! :)
  // TODO: if making patches is too inefficient, maybe we can just do this
  // directly (manually) on the array.
  cv::Mat patch = image_data.GetCroppedPatch(
      0, pixel_index, cv::Size(patch_radius, patch_radius));
  //   OR (since building the image data may be inefficient)
  // cv::Mat patch = (build manually from array values);

  // Apply each degradation operator on the patch. The patch gets transformed
  // and resized after every operator.
  for (const auto& degradation_operator : degradation_operators_) {
    patch = degradation_operator->ApplyToPatch(
        patch, image_index, channel_index, pixel_index);
  }

  // The resulting patch should be just a single pixel.
  // TODO: implement for all of the operators and use this return.
  // CHECK(patch.size() == cv::Size(1, 1));
  // return patch.at<double>(0);

  // TODO: implement for real! This is VERY BAD.
  ImageData degraded_image = ApplyToImage(image_data, image_index);
  degraded_image.ResizeImage(image_data.GetImageSize());
  return degraded_image.GetPixelValue(channel_index, pixel_index);
}

cv::Mat ImageModel::GetModelMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_operators = degradation_operators_.size();
  CHECK_GT(num_operators, 0)
      << "Cannot build a model matrix with no degradation operators.";

  cv::Mat model_matrix =
      degradation_operators_[0]->GetOperatorMatrix(image_size, index);
  for (int i = 1; i < num_operators; ++i) {
    const cv::Mat next_matrix =
        degradation_operators_[i]->GetOperatorMatrix(image_size, index);
    // TODO: multiplication doesn't work with this SparseMat representation.
    model_matrix = next_matrix * model_matrix;
  }
  return model_matrix;
}

}  // namespace super_resolution
