#include "solvers/map_solver.h"

#include <iostream>
#include <vector>

#include "image/image_data.h"
#include "solvers/map_cost_function.h"

#include "opencv2/core/core.hpp"

#include "ceres/ceres.h"

namespace super_resolution {

// TODO: comments.
class ImageProcessingCallback : public ceres::IterationCallback {
 public:
  ceres::CallbackReturnType operator() (
      const ceres::IterationSummary& summary) {

    // TODO: Estimate ^Y_k = image_model.ApplyModel(X) for each k for the next
    // iteration.

    return ceres::SOLVER_CONTINUE;
  }
};

ImageData MapSolver::Solve(const ImageData& initial_estimate) const {
  const int num_observations = low_res_images_.size();
  CHECK(num_observations > 0) << "Cannot solve with 0 low-res images.";

  ImageData estimated_image = initial_estimate;
  const cv::Size image_size = initial_estimate.GetImageSize();

  // Upsample all of the observed low-resolution images to the size of the
  // high-resolution image so we can compare pixels directly.
  std::vector<ImageData> observations;
  for (const ImageData& low_res_image : low_res_images_) {
    ImageData observation = low_res_image;
    observation.ResizeImage(image_size, cv::INTER_AREA);
    observations.push_back(observation);
  }

  // 3. (r) Initialize w_i = 1.

  // 4. Iterate:
  //    a. set est. ^Y_k = image_model_.ApplyModel(X) in low_res_predictions
  //    b. run data term LS between every pixel i of Y_k(i) and X(i)
  //    c. (r) compute regularization at every pixel in estimated X
  //        THIS NEEDS TO BE DONE right after data fidelity estimate is
  //        computed because the regularization is applied to those changes.
  //        Right when regularization is computed, simultaneously compute the
  //        weight w_i.
  //        >>> It might be worth doing this in the regularization cost
  //            function directly.

  // 2. For each frame, apply the image model and naively scale it up to the
  // size of the HR image.
  // TODO: what type of interpolation to use for the scaling? cv::INTER_AREA?
  // TODO: implement this is a callback function/object.
  const int num_frames = low_res_images_.size();
  std::vector<ImageData> low_res_predictions;
  for (int i = 0; i < num_frames; ++i) {
    low_res_predictions.push_back(image_model_.ApplyModel(estimated_image, i));
    // TODO: rescale all of these LR predictions to match the HR size.
  }

  const int num_channels = low_res_images_[0].GetNumChannels();
  const int num_images = low_res_images_.size();
  const int num_pixels = low_res_images_[0].GetNumPixels();

  ceres::Problem problem;
  // TODO: currently solves independently for each channel.
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    for (int image_index = 0; image_index < num_images; ++image_index) {
      for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        ceres::CostFunction* cost_function = MapCostFunction::Create(
            &low_res_predictions, image_index, channel_index, pixel_index);
        problem.AddResidualBlock(
            cost_function,
            NULL,  // basic loss
            estimated_image.GetMutableDataPointer(channel_index, pixel_index));
      }
    }
  }

  // TODO: handle regularization.

  // Set the solver options. TODO: figure out what these should be.
  ceres::Solver::Options options;
  // options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  return estimated_image;
}

}  // namespace super_resolution
