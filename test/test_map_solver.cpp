#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "image/image_data.h"
#include "image_model/blur_module.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "solvers/irls_map_solver.h"
#include "solvers/tv_regularizer.h"
#include "util/test_util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using super_resolution::ImageData;
using super_resolution::test::AreMatricesEqual;

using testing::_;
using testing::ContainerEq;
using testing::DoubleEq;
using testing::ElementsAre;
using testing::Matcher;
using testing::Return;

constexpr bool kPrintSolverOutput = true;
constexpr double kSolverResultErrorTolerance = 0.001;
constexpr double kDerivativeErrorTolerance = 0.000001;
static const super_resolution::MapSolverOptions kDefaultSolverOptions;

// Small image (icon size):
// NOTE: this image cannot exceed 30x30 because of limitations with computing
// the degradation matrices.
static const std::string kTestIconPath = "../test_data/fb.png";

// Bigger image for testing:
static const std::string kTestImagePath = "../test_data/goat.jpg";

class MockRegularizer : public super_resolution::Regularizer {
 public:
  // Handle super constructor, since we don't need the image_size_ field.
  MockRegularizer() : super_resolution::Regularizer(cv::Size(0, 0)) {}

  MOCK_CONST_METHOD1(
      ApplyToImage, std::vector<double>(const double* image_data));

  MOCK_CONST_METHOD2(
      GetDerivatives, std::vector<double>(
          const double* image_data,
          const std::vector<double> partial_const_terms));
};

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
  const int downsampling_scale = 2;
  super_resolution::ImageModel image_model(downsampling_scale);

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
      new super_resolution::DownsamplingModule(
          downsampling_scale, cv::Size(4, 4)));
  image_model.AddDegradationOperator(std::move(downsampling_module));

  const cv::Mat ground_truth_matrix = (cv::Mat_<double>(4, 4)
    << 0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0,
       0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0);
  const ImageData ground_truth_image(ground_truth_matrix);

  /* Verify that the image model produces the correct LR observations. */

  // Create the solver for the model and low-res images.
  super_resolution::IrlsMapSolver solver(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);

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

// Tests on a small icon (real image) and compares the solver result to the
// mathematical derivation result.
TEST(MapSolver, RealIconDataTest) {
  const cv::Mat image = cv::imread(kTestIconPath, CV_LOAD_IMAGE_GRAYSCALE);
  const ImageData ground_truth(image);
  const cv::Size image_size = ground_truth.GetImageSize();

  // Build the image model. 2x downsampling.
  const int downsampling_scale = 2;
  super_resolution::ImageModel image_model(downsampling_scale);

  // Motion.
  const super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(1, 0),
    super_resolution::MotionShift(0, 1),
    super_resolution::MotionShift(1, 1)
  });
  std::unique_ptr<super_resolution::DegradationOperator> motion_module(
      new super_resolution::MotionModule(motion_shift_sequence));
  // Save the matrix for image 0 to make sure the correct matrix operations are
  // being performed.
  const cv::Mat motion_matrix = motion_module->GetOperatorMatrix(image_size, 0);
  image_model.AddDegradationOperator(std::move(motion_module));

  // Downsampling.
  std::unique_ptr<super_resolution::DegradationOperator> downsampling_module(
      new super_resolution::DownsamplingModule(downsampling_scale, image_size));
  const cv::Mat downsampling_matrix =
      downsampling_module->GetOperatorMatrix(image_size, 0);
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
  super_resolution::IrlsMapSolver solver(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  const ImageData solver_result = solver.Solve(initial_estimate);

  // Compare to a solution using the matrix formulation.
  const cv::Mat A1 = image_model.GetModelMatrix(image_size, 0);
  const cv::Mat A2 = image_model.GetModelMatrix(image_size, 1);
  const cv::Mat A3 = image_model.GetModelMatrix(image_size, 2);
  const cv::Mat A4 = image_model.GetModelMatrix(image_size, 3);

  // Make sure we're getting the right matrices.
  const cv::Mat expected_A1 = downsampling_matrix * motion_matrix;
  EXPECT_TRUE(AreMatricesEqual(A1, expected_A1));

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

  const cv::Mat Zinv = Z.inv(cv::DECOMP_SVD);
  cv::Mat matrix_result = Zinv * b;
  matrix_result = matrix_result.reshape(1, image_size.height);

  // Compare the results, but crop out one pixel from the edges to avoid
  // comparing potentially different border handling methods.
  const cv::Rect region_of_interest(1, 1, 26, 26);
  const cv::Mat ground_truth_mat = ground_truth.GetChannelImage(0);
  const cv::Mat cropped_ground_truth = ground_truth_mat(region_of_interest);
  const cv::Mat solver_result_mat = solver_result.GetChannelImage(0);
  const cv::Mat cropped_solver_result = solver_result_mat(region_of_interest);
  const cv::Mat cropped_matrix_result = matrix_result(region_of_interest);
  EXPECT_TRUE(AreMatricesEqual(
      cropped_matrix_result,
      cropped_ground_truth,
      kSolverResultErrorTolerance));
  EXPECT_TRUE(AreMatricesEqual(
      cropped_solver_result,
      cropped_ground_truth,
      kSolverResultErrorTolerance));

  /*const cv::Size disp_size(840, 840);
  ImageData disp_lr_1 = low_res_images[0];
  disp_lr_1.ResizeImage(disp_size);
  cv::imshow("upsampled lr 1", disp_lr_1.GetVisualizationImage());

  ImageData disp_matrix_result(matrix_result);
  disp_matrix_result.ResizeImage(disp_size);
  cv::imshow("Matrix Result", disp_matrix_result.GetVisualizationImage());

  ImageData disp_solver_result = solver_result;
  disp_solver_result.ResizeImage(disp_size);
  cv::imshow("Solver Result", disp_solver_result.GetVisualizationImage());

  ImageData disp_ground_truth = ground_truth;
  disp_ground_truth.ResizeImage(disp_size);
  cv::imshow("Ground Truth", disp_ground_truth.GetVisualizationImage());

  cv::waitKey(0);*/

  // TODO: multichannel test
}

// This test is intended to test the solver's efficiency. It make take a
// little while....
TEST(MapSolver, RealBigImageTest) {
  const cv::Mat image = cv::imread(kTestImagePath, CV_LOAD_IMAGE_GRAYSCALE);
  ImageData ground_truth(image);
  ground_truth.ResizeImage(cv::Size(840, 840));

  // Build the image model. 2x downsampling.
  const int downsampling_scale = 2;
  super_resolution::ImageModel image_model(downsampling_scale);

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

  std::unique_ptr<super_resolution::DegradationOperator> downsampling_module(
      new super_resolution::DownsamplingModule(
          downsampling_scale, ground_truth.GetImageSize()));
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
  super_resolution::IrlsMapSolver solver(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  // TODO: it takes too long (infeasibly long). This needs to be way more
  // scalable.
  // TODO: run some analysis on the code and find out where all the memory is
  // going and where how to make the whole thing way more efficient.
  const ImageData solver_result = solver.Solve(initial_estimate);

  const cv::Size disp_size(840, 840);
  ImageData disp_lr_1 = low_res_images[0];
  disp_lr_1.ResizeImage(disp_size);
  cv::imshow("Upsampled LR #1", disp_lr_1.GetVisualizationImage());

  ImageData disp_ground_truth = ground_truth;
  disp_ground_truth.ResizeImage(disp_size);
  cv::imshow("Ground Truth", disp_ground_truth.GetVisualizationImage());

  ImageData disp_result = solver_result;
  disp_result.ResizeImage(disp_size);
  cv::imshow("Solver Result", disp_result.GetVisualizationImage());

  cv::waitKey(0);
}

// This test uses the full image formation model, including blur, and attempts
// to reconstruct the image using total variation regularization. The
// reconstructed image should not be perfect, but should be close enough.
TEST(MapSolver, RegularizationTest) {
  const cv::Mat image = cv::imread(kTestIconPath, CV_LOAD_IMAGE_GRAYSCALE);
  const ImageData ground_truth(image);
  const cv::Size image_size = ground_truth.GetImageSize();

  // Build the image model. 3x downsampling.
  const int downsampling_scale = 2;  // TODO: 3x
  super_resolution::ImageModel image_model(downsampling_scale);

  // Motion.
  const super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(0, 1),
    //super_resolution::MotionShift(0, 2),
    super_resolution::MotionShift(1, 0),
    super_resolution::MotionShift(1, 1),
    //super_resolution::MotionShift(1, 2),
    //super_resolution::MotionShift(2, 0),
    //super_resolution::MotionShift(2, 1),
    //super_resolution::MotionShift(2, 2)
  });
  std::unique_ptr<super_resolution::DegradationOperator> motion_module(
      new super_resolution::MotionModule(motion_shift_sequence));
  image_model.AddDegradationOperator(std::move(motion_module));

  // Blur.
  std::unique_ptr<super_resolution::DegradationOperator> blur_module(
      new super_resolution::BlurModule(3, 3));
  image_model.AddDegradationOperator(std::move(blur_module));

  // Downsampling.
  std::unique_ptr<super_resolution::DegradationOperator> downsampling_module(
      new super_resolution::DownsamplingModule(downsampling_scale, image_size));
  image_model.AddDegradationOperator(std::move(downsampling_module));

  // Generate the low-res images using the image model.
  const int num_images = motion_shift_sequence.GetNumMotionShifts();
  std::vector<ImageData> low_res_images;
  for (int i = 0; i < num_images; ++i) {
    const super_resolution::ImageData low_res_image =
        image_model.ApplyToImage(ground_truth, i);
    low_res_images.push_back(low_res_image);
  }

  // Set the initial estimate as the upsampling of the referece image, in this
  // case low_res_images[0], since it has no motion shift.
  ImageData initial_estimate = low_res_images[0];
  initial_estimate.ResizeImage(2, cv::INTER_LINEAR);  // bilinear 2x upsampling

  // Create the solver and attempt to solve.
  //solver_options.use_numerical_differentiation = true;
  super_resolution::IrlsMapSolver solver(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  // Add regularizer.
  std::unique_ptr<super_resolution::TotalVariationRegularizer> regularizer(
      new super_resolution::TotalVariationRegularizer(image_size));
  solver.AddRegularizer(std::move(regularizer), 0.01);
  // Solve.
  const ImageData solver_result = solver.Solve(initial_estimate);

  const cv::Size disp_size(840, 840);
  ImageData disp_ground_truth = ground_truth;
  disp_ground_truth.ResizeImage(disp_size);
  cv::imshow("Ground Truth", disp_ground_truth.GetVisualizationImage());

  ImageData disp_result = solver_result;
  disp_result.ResizeImage(disp_size);
  cv::imshow("Solver Result", disp_result.GetVisualizationImage());

  cv::waitKey(0);
}

// Verifies that the IrlsMapSolver computes the correct data term residuals.
TEST(MapSolver, IrlsComputeDataTerm) {
  const cv::Size image_size(3, 3);
  const cv::Mat lr_channel_1 = (cv::Mat_<double>(3, 3)
      << 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5,
         0.5, 0.5, 0.5);
  const cv::Mat lr_channel_2 = (cv::Mat_<double>(3, 3)
      << 1.0,  0.5,  0.0,
         0.25, 0.5,  0.75,
         1.0,  0.0,  1.0);
  const cv::Mat lr_channel_3 = (cv::Mat_<double>(3, 3)
      << 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0,
         1.0, 1.0, 1.0);

  super_resolution::ImageData lr_image_data_1;
  lr_image_data_1.AddChannel(lr_channel_1);
  lr_image_data_1.AddChannel(lr_channel_2);
  super_resolution::ImageData lr_image_data_2(lr_channel_3);
  const std::vector<super_resolution::ImageData> low_res_images = {
    lr_image_data_1,  // Image with 2 channels
    lr_image_data_2   // Image with 1 channel
  };

  // Empty image model (does nothing) and no regularizer.
  const super_resolution::ImageModel empty_image_model(1);
  super_resolution::IrlsMapSolver irls_map_solver(
      kDefaultSolverOptions,
      empty_image_model,
      low_res_images);

  const double hr_pixel_values[9] = {
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5
  };

  // (Image 1, Channel 1) and hr pixels are identical, so expect all zeros.
  const auto& residual_sum_and_gradient_1 =
      irls_map_solver.ComputeDataTermAnalyticalDiff(0, 0, hr_pixel_values);
  EXPECT_EQ(residual_sum_and_gradient_1.first, 0);

  // (Image 1, Channel 2) residuals should be different at each pixel.
  const auto& residual_sum_and_gradient_2 =
      irls_map_solver.ComputeDataTermAnalyticalDiff(0, 1, hr_pixel_values);
  const std::vector<double> expected_residuals = {
      -0.5,  0.0,  0.5,
       0.25, 0.0, -0.25,
      -0.5,  0.5, -0.5
  };
  double expected_residual_sum = 0.0;
  for (const double residual : expected_residuals) {
    expected_residual_sum += (residual * residual);
  }
  EXPECT_EQ(residual_sum_and_gradient_2.first, expected_residual_sum);

  // (Image 2, channel 1) ("channel_3") should all be -0.5.
  const auto& residual_sum_and_gradient_3 =
      irls_map_solver.ComputeDataTermAnalyticalDiff(1, 0, hr_pixel_values);
  expected_residual_sum = 0.0;
  for (int i = 0; i < 9; ++i) {
    expected_residual_sum += (-0.5 * -0.5);
  }
  EXPECT_EQ(residual_sum_and_gradient_3.first, expected_residual_sum);

  // TODO: Mock the ImageModel and make sure the residuals are computed
  // correctly if the HR image is degraded first.
}

TEST(MapSolver, IrlsComputeRegularization) {
  // Mocked Regularizer.
  std::unique_ptr<MockRegularizer> mock_regularizer(new MockRegularizer());
  const double image_data[5] = {1, 2, 3, 4, 5};
  const std::vector<double> residuals = {1, 2, 3, 4, 5};
  EXPECT_CALL(*mock_regularizer, ApplyToImage(image_data))
      .Times(3)  // 3 calls: compute residuals, update weights, compute again.
      .WillRepeatedly(Return(residuals));
  EXPECT_CALL(*mock_regularizer, GetDerivatives(_, _))
      .Times(2)  // 2 calls: once for each ComputeRegularizationAnalyticalDiff.
      .WillRepeatedly(Return(residuals));

  const std::vector<super_resolution::ImageData> low_res_images = {
    ImageData(image_data, cv::Size(5, 1))  // Ignored image for this test.
  };
  const super_resolution::ImageModel empty_image_model(1);
  const double regularization_parameter = 0.5;
  super_resolution::IrlsMapSolver irls_map_solver(
      kDefaultSolverOptions,
      empty_image_model,
      low_res_images);
  irls_map_solver.AddRegularizer(
      std::move(mock_regularizer), regularization_parameter);

  // Expected residuals should be the residuals returned by the mocked
  // Regularizer times the regularization parameter and the square root of the
  // respective weights, which are all 1.0 to begin with.
  const std::vector<double> expected_residuals_1 = {
      residuals[0] * regularization_parameter,
      residuals[1] * regularization_parameter,
      residuals[2] * regularization_parameter,
      residuals[3] * regularization_parameter,
      residuals[4] * regularization_parameter,
  };
  double expected_residual_sum = 0.0;
  for (const double residual : expected_residuals_1) {
    expected_residual_sum += (residual * residual);
  }
  const auto& residual_sum_and_gradient_1 =
      irls_map_solver.ComputeRegularizationAnalyticalDiff(image_data);
  EXPECT_EQ(residual_sum_and_gradient_1.first, expected_residual_sum);

  // Update weights and test again. The weights are expected to be updated as
  // follows:
  //   w = 1.0 / sqrt(residual)
  // so, given residuals [1, 2, 3, 4, 5]:
  //   w0 = 1.0 / 1.0 ~= 1.0
  //   w1 = 1.0 / 2.0 ~= 0.5
  //   w2 = 1.0 / 3.0 ~= 0.333333333
  //   w3 = 1.0 / 4.0 ~= 0.25
  //   w4 = 1.0 / 5.0 ~= 0.2
  //
  // TODO: test with updated weights for a non-L1 norm regularizer.
  irls_map_solver.UpdateIrlsWeights(image_data);

  // Now expect the residuals to be multiplied by the regularization parameter
  // and the square root of the newly computed weights.
  const std::vector<double> expected_residuals_2 = {
    residuals[0] * regularization_parameter * sqrt(1.0 / 1.0),
    residuals[1] * regularization_parameter * sqrt(1.0 / 2.0),
    residuals[2] * regularization_parameter * sqrt(1.0 / 3.0),
    residuals[3] * regularization_parameter * sqrt(1.0 / 4.0),
    residuals[4] * regularization_parameter * sqrt(1.0 / 5.0)
  };
  expected_residual_sum = 0.0;
  for (const double residual : expected_residuals_2) {
    expected_residual_sum += (residual * residual);
  }
  const auto& residual_sum_and_gradient_2 =
      irls_map_solver.ComputeRegularizationAnalyticalDiff(image_data);
  EXPECT_EQ(residual_sum_and_gradient_2.first, expected_residual_sum);
}

// This test checks the validity of the differentiation methods, specifically
// comparing results of analytical differentiation to numerical differentiation
// to help debug any implementations of manually computing gradients.
TEST(MapSolver, IrlsDifferentiationTest) {
  // TODO: finish this test or remove it.

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
  const std::vector<super_resolution::ImageData> low_res_images {
    super_resolution::ImageData(lr_image_1),
    super_resolution::ImageData(lr_image_2),
    super_resolution::ImageData(lr_image_3),
    super_resolution::ImageData(lr_image_4)
  };

  // Create the image model.
  const cv::Size image_size(4, 4);
  const int downsampling_scale = 2;
  super_resolution::ImageModel image_model(downsampling_scale);

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
      new super_resolution::DownsamplingModule(downsampling_scale, image_size));
  image_model.AddDegradationOperator(std::move(downsampling_module));

  /* Test differentiation with TV regularizer. */

  const int num_variables = 16;
  const double estimated_data[num_variables] = {
    0.0, 0.1, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7,
    0.8, 0.9, 1.0, 0.5,
    0.1, 0.3, 0.7, 0.9
  };

  // Verify by checking that numerical (finite difference) differentiation
  // produces the same results as the implemented analytical differentiation.
  double analytical_gradient[num_variables];
  double numerical_gradient[num_variables];

  const int num_images = low_res_images.size();
  const double numerical_diff_step_size = 1.0e-6;

  // Loop over multiple regularization parameter values.
  for (int i = 0; i < 2; ++i) {
    super_resolution::IrlsMapSolver irls_map_solver(
        kDefaultSolverOptions,
        image_model,
        low_res_images);
    const double regularization_parameter = 1 * i;
    std::unique_ptr<super_resolution::Regularizer> tv_regularizer(
        new super_resolution::TotalVariationRegularizer(image_size));
    irls_map_solver.AddRegularizer(
        std::move(tv_regularizer), regularization_parameter);

    const int num_pixels = irls_map_solver.GetNumPixels();
    double residual_sum = 0;
    double gradient[num_pixels];
    std::fill(gradient, gradient + num_pixels, 0);

    for (int image_index = 0; image_index < num_images; ++image_index) {
      const auto& residual_sum_and_gradient =
          irls_map_solver.ComputeDataTermAnalyticalDiff(
              image_index, 0, estimated_data);  // channel 0
      residual_sum += residual_sum_and_gradient.first;
      for (int i = 0; i < num_pixels; ++i) {
        gradient[i] += residual_sum_and_gradient.second[i];
      }
    }

    const auto& residual_sum_and_gradient =
        irls_map_solver.ComputeRegularizationAnalyticalDiff(estimated_data);
    residual_sum += residual_sum_and_gradient.first;
    for (int i = 0; i < num_pixels; ++i) {
      gradient[i] += residual_sum_and_gradient.second[i];
    }

    std::cout << "Objective = " << residual_sum << std::endl;

    //// Compute numerical finite difference gradient.
    //for (int i = 0; i < num_variables; ++i) {
    //  // Copy the data and pertrube the ith variable by a small amount.
    //  double estimated_data_diff[num_variables];
    //  std::copy(
    //      estimated_data, estimated_data + num_variables, estimated_data_diff);
    //  estimated_data_diff[i] += numerical_diff_step_size;
    //  // Compute the numerical derivative at the ith variable.
    //  const double diff_objective_value =
    //      irls_cost_processor.ComputeObjectiveFunction(estimated_data_diff);
    //  numerical_gradient[i] =
    //      (diff_objective_value - objective_value) / numerical_diff_step_size;
    //}

    //// Normalize both gradients.
    //double analytical_gradient_norm = 0;
    //double numerical_gradient_norm = 0;
    //for (int i = 0; i < num_variables; ++i) {
    //  analytical_gradient_norm += analytical_gradient[i] * analytical_gradient[i];
    //  numerical_gradient_norm += numerical_gradient[i] * numerical_gradient[i];
    //}
    //analytical_gradient_norm = sqrt(analytical_gradient_norm);
    //numerical_gradient_norm = sqrt(numerical_gradient_norm);
    //for (int i = 0; i < num_variables; ++i) {
    //  analytical_gradient[i] /= analytical_gradient_norm;
    //  numerical_gradient[i] /= numerical_gradient_norm;
    //}

    //// Compare the two results.
    //for (int i = 0; i < num_variables; ++i) {
    //  std::cout << "Gradient @ " << i << std::endl;
    //  std::cout << "   Analytical: " << analytical_gradient[i] << std::endl;
    //  std::cout << "   Numerical:  " << numerical_gradient[i] << std::endl;
    //  // EXPECT_NEAR(
    //  //     analytical_gradient[i],
    //  //     numerical_gradient[i],
    //  //     kDerivativeErrorTolerance);
    //}
  }
}
