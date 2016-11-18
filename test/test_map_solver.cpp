#include <iostream> // TODO REMOVE
#include <vector>

#include "image/image_data.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "solvers/map_solver.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using super_resolution::ImageData;

constexpr double kSolverResultErrorTolerance = 0.001;

static const std::string kTestIconPath = "../test_data/fb.png";

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
  super_resolution::MotionShiftSequence motion_shift_sequence({
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

  const cv::Mat ground_truth_matrix = (cv::Mat_<double>(4, 4)
    << 0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0,
       0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0);
  const ImageData ground_truth_image(ground_truth_matrix);

  /* Verify that the image model produces the correct LR observations. */

  // Create the solver for the model and low-res images.
  super_resolution::MapSolver solver(image_model, low_res_images);

  // Create the high-res initial estimate.
  const cv::Mat initial_estimate_matrix = (cv::Mat_<double>(4, 4)
      << 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0);
  ImageData initial_estimate(initial_estimate_matrix);

  /* Verify solver gets a near-perfect solution for this trivial case. */

  ImageData result = solver.Solve(initial_estimate);
  for (int pixel_index = 0; pixel_index < 16; ++pixel_index) {
    EXPECT_NEAR(
        result.GetPixelValue(0, pixel_index),
        ground_truth_image.GetPixelValue(0, pixel_index),
        kSolverResultErrorTolerance);
  }
}

TEST(MapSolver, RealIconDataTest) {
  const cv::Mat image = cv::imread(kTestIconPath, CV_LOAD_IMAGE_GRAYSCALE);
  const ImageData ground_truth(image);

  // Build the image model.
  super_resolution::ImageModel image_model;

  // Motion.
  super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(-1, 0),
    super_resolution::MotionShift(0, -1),
    super_resolution::MotionShift(-1, -1)
  });
  std::unique_ptr<super_resolution::DegradationOperator> motion_module(
      new super_resolution::MotionModule(motion_shift_sequence));
  image_model.AddDegradationOperator(std::move(motion_module));

  // Blur. TODO!
  // image_model.AddDegradationOperator(blur_module);

  // 2x downsampling.
  std::unique_ptr<super_resolution::DegradationOperator> downsampling_module(
      new super_resolution::DownsamplingModule(2));
  image_model.AddDegradationOperator(std::move(downsampling_module));

  // Generate the low-res images using the image model.
  std::vector<ImageData> low_res_images {
    image_model.ApplyModel(ground_truth, 0),
    image_model.ApplyModel(ground_truth, 1),
    image_model.ApplyModel(ground_truth, 2),
    image_model.ApplyModel(ground_truth, 3)
  };

  // Set the initial estimate as the upsampling of the referece image, in this
  // case lr_image_1, since it has no motion shift.
  ImageData initial_estimate = low_res_images[0];
  initial_estimate.ResizeImage(2, cv::INTER_LINEAR);  // bilinear 2x upsampling
  
  // Create the solver and attempt to solve.
  //TODO put back
  //super_resolution::MapSolver solver(image_model, low_res_images);
  //ImageData result = solver.Solve(initial_estimate);

  // Compare to a solution using the matrix formulation.
  const cv::Size image_size = ground_truth.GetImageSize();
  cv::Mat A1 = image_model.GetModelMatrix(image_size, 0);
  cv::Mat A2 = image_model.GetModelMatrix(image_size, 1);
  cv::Mat A3 = image_model.GetModelMatrix(image_size, 2);
  cv::Mat A4 = image_model.GetModelMatrix(image_size, 3);
  std::cout << "Get Model matrices done" << std::endl;
  std::cout << A1 << std::endl;

  // Linear system: x = Z^ * b, and thus Zx = b.
  // x = sum(A'A)^ * sum(A'y) (' is transpose, ^ is inverse).
  cv::Mat Z = A1.t() * A1;
  Z += A2.t() * A2;
  Z += A3.t() * A3;
  Z += A4.t() * A4;
  // TODO: Z += regularization term
  std::cout << "Compute Z done" << std::endl;
  std::cout << Z << std::endl;

  const int num_pixels = (image_size.width * image_size.height) / 4;
  cv::Mat b =
      A1.t() * low_res_images[0].GetChannelImage(0).reshape(1, num_pixels);
  b +=  A2.t() * low_res_images[1].GetChannelImage(0).reshape(1, num_pixels);
  b +=  A3.t() * low_res_images[2].GetChannelImage(0).reshape(1, num_pixels);
  b +=  A4.t() * low_res_images[3].GetChannelImage(0).reshape(1, num_pixels);
  std::cout << "Compute b done" << std::endl;
  std::cout << b << std::endl;

  cv::Mat x = Z.inv() * b;
  std::cout << "Compute x done" << std::endl;
  x = x.reshape(1, image_size.height);
  std::cout << x << std::endl;

  ImageData disp_x(x);
  disp_x.ResizeImage(cv::Size(1024, 1024));
  cv::imshow("x", disp_x.GetVisualizationImage());

//  ImageData disp_lr_1 = low_res_images[0];
//  disp_lr_1.ResizeImage(cv::Size(1024, 1024));
//  cv::imshow("upsampled lr 1", disp_lr_1.GetVisualizationImage());
//
//  ImageData disp_ground_truth = ground_truth;
//  disp_ground_truth.ResizeImage(cv::Size(1024, 1024));
//  cv::imshow("ground truth", disp_ground_truth.GetVisualizationImage());
//
//  ImageData disp_result = result;
//  disp_result.ResizeImage(cv::Size(1024, 1024));
//  cv::imshow("super-resolved", disp_result.GetVisualizationImage());

  cv::waitKey(0);
}
