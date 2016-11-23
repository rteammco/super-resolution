#include <iostream> // TODO REMOVE
#include <memory>
#include <string>
#include <vector>

#include "image/image_data.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "solvers/map_solver.h"
#include "util/test_util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using super_resolution::ImageData;
using super_resolution::test::AreMatricesEqual;

constexpr bool kPrintSolverOutput = true;
constexpr double kSolverResultErrorTolerance = 0.001;

// Small image (icon size):
// NOTE: this image cannot exceed 30x30 because of limitations with computing
// the degradation matrices.
static const std::string kTestIconPath = "../test_data/fb.png";

// Bigger image for testing:
static const std::string kTestImagePath = "../test_data/goat.jpg";

// Tests the solver on small, "perfect" data to make sure it works as expected.
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
  super_resolution::MapSolverOptions solver_options;
  solver_options.regularization_parameter = 0.0;
  super_resolution::MapSolver solver(
      solver_options, image_model, low_res_images, kPrintSolverOutput);

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

  // TODO: multichannel image test
}

// Tests on a small icon (real image) and compares the Ceres solver result to
// the mathematical derivation result.
TEST(MapSolver, RealIconDataTest) {
  const cv::Mat image = cv::imread(kTestIconPath, CV_LOAD_IMAGE_GRAYSCALE);
  const ImageData ground_truth(image);

  // Build the image model.
  super_resolution::ImageModel image_model;

  // Motion.
  super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(1, 0),
    super_resolution::MotionShift(0, 1),
    super_resolution::MotionShift(1, 1)
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
  const std::vector<ImageData> low_res_images {
    image_model.ApplyToImage(ground_truth, 0),
    image_model.ApplyToImage(ground_truth, 1),
    image_model.ApplyToImage(ground_truth, 2),
    image_model.ApplyToImage(ground_truth, 3)
  };

  // Set the initial estimate as the upsampling of the referece image, in this
  // case lr_image_1, since it has no motion shift.
  ImageData initial_estimate = low_res_images[0];
  initial_estimate.ResizeImage(2, cv::INTER_LINEAR);  // bilinear 2x upsampling

  // Create the solver and attempt to solve.
  super_resolution::MapSolverOptions solver_options;
  solver_options.regularization_parameter = 0.0;
  super_resolution::MapSolver solver(
      solver_options, image_model, low_res_images, kPrintSolverOutput);
  const ImageData solver_result = solver.Solve(initial_estimate);

  // Compare to a solution using the matrix formulation.
  const cv::Size image_size = ground_truth.GetImageSize();
  cv::Mat A1 = image_model.GetModelMatrix(image_size, 0);
  cv::Mat A2 = image_model.GetModelMatrix(image_size, 1);
  cv::Mat A3 = image_model.GetModelMatrix(image_size, 2);
  cv::Mat A4 = image_model.GetModelMatrix(image_size, 3);

  // Linear system: x = Z^ * b, and thus Zx = b.
  // x = sum(A'A)^ * sum(A'y) (' is transpose, ^ is inverse).
  cv::Mat Z = A1.t() * A1;
  Z += A2.t() * A2;
  Z += A3.t() * A3;
  Z += A4.t() * A4;
  // TODO: Z += regularization term

  const int num_pixels = (image_size.width * image_size.height) / 4;
  cv::Mat b =
      A1.t() * low_res_images[0].GetChannelImage(0).reshape(1, num_pixels);
  b +=  A2.t() * low_res_images[1].GetChannelImage(0).reshape(1, num_pixels);
  b +=  A3.t() * low_res_images[2].GetChannelImage(0).reshape(1, num_pixels);
  b +=  A4.t() * low_res_images[3].GetChannelImage(0).reshape(1, num_pixels);

  cv::Mat Zinv = Z.inv(cv::DECOMP_SVD);
  cv::Mat matrix_result = Zinv * b;
  matrix_result = matrix_result.reshape(1, image_size.height);

/*  const cv::Size disp_size(840, 840);
  ImageData disp_x(matrix_result);
  disp_x.ResizeImage(disp_size);
  cv::imshow("x", disp_x.GetVisualizationImage());

  ImageData disp_lr_1 = low_res_images[0];
  disp_lr_1.ResizeImage(disp_size);
  cv::imshow("upsampled lr 1", disp_lr_1.GetVisualizationImage());

  ImageData disp_ground_truth = ground_truth;
  disp_ground_truth.ResizeImage(disp_size);
  cv::imshow("ground truth", disp_ground_truth.GetVisualizationImage());

  ImageData disp_result = solver_result;
  disp_result.ResizeImage(disp_size);
  cv::imshow("super-resolved", disp_result.GetVisualizationImage());

  cv::waitKey(0);
  */

  // TODO: multichannel test
}

// This test is intended to test the solver's efficiency. It make take a
// while...
// TODO: fix this test and run it after implementing ImageModel::ApplyToPixel.
TEST(MapSolver, RealBigImageTest) {
  const cv::Mat image = cv::imread(kTestImagePath, CV_LOAD_IMAGE_GRAYSCALE);
  ImageData ground_truth(image);
  ground_truth.ResizeImage(cv::Size(100, 100));

  std::cout << "Loaded image." << std::endl;

  // Build the image model.
  super_resolution::ImageModel image_model;

  // Motion.
  super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(1, 0),
    super_resolution::MotionShift(0, 1),
    super_resolution::MotionShift(1, 1)
  });
  std::unique_ptr<super_resolution::DegradationOperator> motion_module(
      new super_resolution::MotionModule(motion_shift_sequence));
  image_model.AddDegradationOperator(std::move(motion_module));

  // 2x downsampling.
  std::unique_ptr<super_resolution::DegradationOperator> downsampling_module(
      new super_resolution::DownsamplingModule(2));
  image_model.AddDegradationOperator(std::move(downsampling_module));

  std::cout << "Built image." << std::endl;

  // Generate the low-res images using the image model.
  const std::vector<ImageData> low_res_images {
    image_model.ApplyToImage(ground_truth, 0),
    image_model.ApplyToImage(ground_truth, 1),
    image_model.ApplyToImage(ground_truth, 2),
    image_model.ApplyToImage(ground_truth, 3)
  };

  // Set the initial estimate as the upsampling of the referece image, in this
  // case lr_image_1, since it has no motion shift.
  ImageData initial_estimate = low_res_images[0];
  initial_estimate.ResizeImage(2, cv::INTER_LINEAR);  // bilinear 2x upsampling

  std::cout << "Created initial estimate." << std::endl;

  // Create the solver and attempt to solve.
  super_resolution::MapSolverOptions solver_options;
  solver_options.regularization_parameter = 0.0;
  super_resolution::MapSolver solver(
      solver_options, image_model, low_res_images, kPrintSolverOutput);
  // TODO: it takes too long (infeasibly long). This needs to be way more
  // scalable.
  // TODO: run some analysis on the code and find out where all the memory is
  // going and where how to make the whole thing way more efficient.
  /*const ImageData solver_result = solver.Solve(initial_estimate);

  std::cout << "Solved." << std::endl;

  const cv::Size disp_size(840, 840);
  ImageData disp_lr_1 = low_res_images[0];
  disp_lr_1.ResizeImage(disp_size);
  cv::imshow("upsampled lr 1", disp_lr_1.GetVisualizationImage());

  ImageData disp_ground_truth = ground_truth;
  disp_ground_truth.ResizeImage(disp_size);
  cv::imshow("ground truth", disp_ground_truth.GetVisualizationImage());

  ImageData disp_result = solver_result;
  disp_result.ResizeImage(disp_size);
  cv::imshow("super-resolved", disp_result.GetVisualizationImage());

  cv::waitKey(0); */
}
