#include <iostream>  // TODO: remove
#include <vector>

#include "image/image_data.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "solvers/map_solver.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::ImageData;

constexpr double kSolverResultErrorTolerance = 0.001;

TEST(MapSolver, SmallDataTest) {
  // Create the low-res test images.
  const cv::Mat lr_image_1 = (cv::Mat_<double>(2, 2)
    << 0.4, 0.4,
       0.4, 0.4);
  const cv::Mat lr_image_2 = (cv::Mat_<double>(2, 2)
    << 0.2, 0.2,
       0.2, 0.2);
  const cv::Mat lr_image_3 = (cv::Mat_<double>(2, 2)
    << 0.0, 0.0,
       0.0, 0.0);
  const cv::Mat lr_image_4 = (cv::Mat_<double>(2, 2)
    << 1.0, 1.0,
       1.0, 1.0);
  std::vector<ImageData> low_res_images {
    ImageData(lr_image_1),
    ImageData(lr_image_2),
    ImageData(lr_image_3),
    ImageData(lr_image_4)
  };

  // Create the image model.
  super_resolution::ImageModel image_model;

  // Add motion:
  super_resolution::MotionShiftSequence motion_shift_sequence;
  motion_shift_sequence.SetMotionSequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(-1, 0),
    super_resolution::MotionShift(0, -1),
    super_resolution::MotionShift(-1, -1)
  });
  std::unique_ptr<super_resolution::DegradationOperator> motion_module(
      new super_resolution::MotionModule(motion_shift_sequence));
  image_model.AddDegradationOperator(std::move(motion_module));

  // Add downsampling:
  std::unique_ptr<super_resolution::DegradationOperator> downsampling_module(
      new super_resolution::DownsamplingModule(2));
  image_model.AddDegradationOperator(std::move(downsampling_module));

  /* Verify that the image model produces the correct LR observations. */

  const cv::Mat ground_truth_matrix = (cv::Mat_<double>(4, 4)
    << 0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0,
       0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0);
  const ImageData ground_truth_image(ground_truth_matrix);
  ImageData lr_simulated_1 = image_model.ApplyModel(ground_truth_image, 0);
  ImageData lr_simulated_2 = image_model.ApplyModel(ground_truth_image, 1);
  ImageData lr_simulated_3 = image_model.ApplyModel(ground_truth_image, 2);
  ImageData lr_simulated_4 = image_model.ApplyModel(ground_truth_image, 3);

  // TODO: remove
  //std::cout << lr_simulated_1.GetChannelImage(0) << std::endl;
  //std::cout << lr_simulated_2.GetChannelImage(0) << std::endl;
  //std::cout << lr_simulated_3.GetChannelImage(0) << std::endl;
  //std::cout << lr_simulated_4.GetChannelImage(0) << std::endl;

  // Create the solver for the model and low-res images.
  super_resolution::MapSolver solver(image_model, low_res_images);

  // Create the high-res initial estimate.
  const cv::Mat initial_estimate_matrix = (cv::Mat_<double>(4, 4)
      << 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0);
      //<< 0.5, 0.5, 0.5, 0.5,
      //   0.5, 0.5, 0.5, 0.5,
      //   0.5, 0.5, 0.5, 0.5,
      //   0.5, 0.5, 0.5, 0.5);
      //<< 1, 1, 1, 1,
      //   1, 1, 1, 1,
      //   1, 1, 1, 1,
      //   1, 1, 1, 1);
  ImageData initial_estimate(initial_estimate_matrix);

  ImageData result = solver.Solve(initial_estimate);

  for (int pixel_index = 0; pixel_index < 16; ++pixel_index) {
    EXPECT_NEAR(
        result.GetPixelValue(0, pixel_index),
        ground_truth_image.GetPixelValue(0, pixel_index),
        kSolverResultErrorTolerance);
  }
  // TODO: remove
  //std::cout << result.GetChannelImage(0) << std::endl;
}
