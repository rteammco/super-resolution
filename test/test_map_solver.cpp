#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "evaluation/peak_signal_to_noise_ratio.h"
#include "image/image_data.h"
#include "image_model/additive_noise_module.h"
#include "image_model/blur_module.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "optimization/btv_regularizer.h"
#include "optimization/irls_map_solver.h"
#include "optimization/tv_regularizer.h"
#include "util/test_util.h"
#include "util/util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using super_resolution::AdditiveNoiseModule;
using super_resolution::BlurModule;
using super_resolution::DownsamplingModule;
using super_resolution::ImageData;
using super_resolution::MotionModule;
using super_resolution::test::AreMatricesEqual;
using super_resolution::util::GetAbsoluteCodePath;

using testing::_;
using testing::ContainerEq;
using testing::DoubleEq;
using testing::ElementsAre;
using testing::Matcher;
using testing::Return;

constexpr bool kDisplaySolverResults = false;
static const cv::Size kDisplayImageSize(840, 840);

constexpr bool kPrintSolverOutput = true;
constexpr double kSolverResultErrorTolerance = 0.001;
constexpr double kDerivativeErrorTolerance = 0.000001;
static const super_resolution::IrlsMapSolverOptions kDefaultSolverOptions;

// Small image (icon size):
// NOTE: this image cannot exceed 30x30 because of limitations with computing
// the degradation matrices.
static const std::string kTestIconPath =
    GetAbsoluteCodePath("test_data/fb.png");

// Bigger image for testing:
static const std::string kTestImagePath =
    GetAbsoluteCodePath("test_data/goat.jpg");

class MockRegularizer : public super_resolution::Regularizer {
 public:
  // Handle super constructor, since we don't need the image_size_ or the
  // num_channels_ fields.
  MockRegularizer() : super_resolution::Regularizer(cv::Size(0, 0), 1) {}

  MOCK_CONST_METHOD1(
      ApplyToImage, std::vector<double>(const double* image_data));

  MOCK_CONST_METHOD2(
      ApplyToImageWithDifferentiation,
      std::pair<std::vector<double>, std::vector<double>>(
          const double* image_data,
          const std::vector<double>& gradient_constants));
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
  const std::vector<cv::Mat> lr_image_matrices = {
    lr_image_1, lr_image_2, lr_image_3, lr_image_4
  };
  std::vector<ImageData> low_res_images;
  for (const cv::Mat& lr_image_matrix : lr_image_matrices) {
    low_res_images.push_back(ImageData(lr_image_matrix));
  }

  // Create the image model.
  super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(-1, 0),
    super_resolution::MotionShift(0, -1),
    super_resolution::MotionShift(-1, -1)
  });
  super_resolution::ImageModelParameters model_parameters;
  model_parameters.scale = 2;
  model_parameters.motion_sequence = motion_shift_sequence;
  const super_resolution::ImageModel image_model =
      super_resolution::ImageModel::CreateImageModel(model_parameters);

  /* Verify solver gets a near-perfect solution for this trivial case. */

  // Expected results:
  const cv::Mat ground_truth_matrix = (cv::Mat_<double>(4, 4)
    << 0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0,
       0.4, 0.2, 0.4, 0.2,
       0.0, 1.0, 0.0, 1.0);
  const ImageData ground_truth_image(ground_truth_matrix);

  // Create the high-res initial estimate.
  const cv::Mat initial_estimate_matrix = (cv::Mat_<double>(4, 4)
      << 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0);
  const ImageData initial_estimate(initial_estimate_matrix);

  // Create the solver for the model and low-res images.
  super_resolution::IrlsMapSolver solver(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  const ImageData result = solver.Solve(initial_estimate);

  EXPECT_TRUE(AreMatricesEqual(
      result.GetChannelImage(0),
      ground_truth_matrix,
      kSolverResultErrorTolerance));

  /* Repeat the same tests, but this time with multiple channels. */

  // Simply replicate the channels for each image.
  const int num_channels = 10;
  std::vector<ImageData> low_res_images_multichannel;
  for (const cv::Mat& lr_image_matrix : lr_image_matrices) {
    ImageData multichannel_image;
    for (int j = 0; j < num_channels; ++j)  {
      multichannel_image.AddChannel(lr_image_matrix);
    }
    low_res_images_multichannel.push_back(multichannel_image);
  }

  // Also replicate channels for the initial estimate and ground truth.
  ImageData ground_truth_image_multichannel;
  ImageData initial_estimate_multichannel;
  for (int i = 0; i < num_channels; ++i) {
    ground_truth_image_multichannel.AddChannel(ground_truth_matrix);
    initial_estimate_multichannel.AddChannel(initial_estimate_matrix);
  }

  // Create the multichannel solver.
  super_resolution::IrlsMapSolver solver_multichannel(
      kDefaultSolverOptions,
      image_model,
      low_res_images_multichannel,
      kPrintSolverOutput);
  const ImageData result_multichannel =
      solver_multichannel.Solve(initial_estimate_multichannel);

  EXPECT_EQ(result_multichannel.GetNumChannels(), num_channels);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    EXPECT_TRUE(AreMatricesEqual(
        result_multichannel.GetChannelImage(channel_index),
        ground_truth_matrix,
        kSolverResultErrorTolerance));
  }
}

// Tests on a small icon (real image) and compares the solver result to the
// mathematical derivation result. This will be a single-channel test since
// it also test the mathematical implementation, which only supports a single
// channel.
TEST(MapSolver, RealIconDataTest) {
  const cv::Mat image = cv::imread(kTestIconPath, CV_LOAD_IMAGE_GRAYSCALE);
  const ImageData ground_truth(image);
  const cv::Size image_size = ground_truth.GetImageSize();

  // Build the image model. 2x downsampling. Create it manually here because we
  // need to get the degradation operator matrices.
  const int downsampling_scale = 2;
  super_resolution::ImageModel image_model(downsampling_scale);

  // Motion.
  const super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(1, 0),
    super_resolution::MotionShift(0, 1),
    super_resolution::MotionShift(1, 1)
  });
  std::shared_ptr<MotionModule> motion_module(
      new MotionModule(motion_shift_sequence));
  // Save the matrix for image 0 to make sure the correct matrix operations are
  // being performed.
  const cv::Mat motion_matrix = motion_module->GetOperatorMatrix(image_size, 0);
  image_model.AddDegradationOperator(motion_module);

  // Downsampling.
  std::shared_ptr<DownsamplingModule> downsampling_module(
      new DownsamplingModule(downsampling_scale));
  const cv::Mat downsampling_matrix =
      downsampling_module->GetOperatorMatrix(image_size, 0);
  image_model.AddDegradationOperator(downsampling_module);

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
  initial_estimate.ResizeImage(2, super_resolution::INTERPOLATE_LINEAR);

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

  if (kDisplaySolverResults) {
    ImageData disp_lr_1 = low_res_images[0];
    disp_lr_1.ResizeImage(kDisplayImageSize);
    cv::imshow("upsampled lr 1", disp_lr_1.GetVisualizationImage());

    ImageData disp_matrix_result(matrix_result);
    disp_matrix_result.ResizeImage(kDisplayImageSize);
    cv::imshow("Matrix Result", disp_matrix_result.GetVisualizationImage());

    ImageData disp_solver_result = solver_result;
    disp_solver_result.ResizeImage(kDisplayImageSize);
    cv::imshow("Solver Result", disp_solver_result.GetVisualizationImage());

    ImageData disp_ground_truth = ground_truth;
    disp_ground_truth.ResizeImage(kDisplayImageSize);
    cv::imshow("Ground Truth", disp_ground_truth.GetVisualizationImage());

    cv::waitKey(0);
  }
}

// This test is intended to test the solver's efficiency as well as its ability
// to handle real-world multichannel images.
TEST(MapSolver, RealBigImageTest) {
  const cv::Mat image = cv::imread(kTestImagePath, CV_LOAD_IMAGE_COLOR);
  ImageData ground_truth(image);
  ground_truth.ResizeImage(cv::Size(840, 840));

  // Build the image model. 2x downsampling.
  super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    super_resolution::MotionShift(1, 0),
    super_resolution::MotionShift(0, 1),
    super_resolution::MotionShift(1, 1)
  });
  super_resolution::ImageModelParameters model_parameters;
  model_parameters.scale = 2;
  model_parameters.motion_sequence = motion_shift_sequence;
  const super_resolution::ImageModel image_model =
      super_resolution::ImageModel::CreateImageModel(model_parameters);

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
  initial_estimate.ResizeImage(2, super_resolution::INTERPOLATE_LINEAR);

  // Create the solver and attempt to solve.
  super_resolution::IrlsMapSolver solver(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  const ImageData solver_result = solver.Solve(initial_estimate);

  const int num_channels = ground_truth.GetNumChannels();
  EXPECT_EQ(solver_result.GetNumChannels(), num_channels);
  for (int channel_index = 0; channel_index < num_channels; ++channel_index) {
    EXPECT_TRUE(super_resolution::test::AreMatricesEqualCroppedBorder(
        solver_result.GetChannelImage(channel_index),
        ground_truth.GetChannelImage(channel_index),
        1,  // Don't check the border pixels (border of size 1).
        kSolverResultErrorTolerance));
  }

  if (kDisplaySolverResults) {
    ImageData disp_lr_1 = low_res_images[0];
    disp_lr_1.ResizeImage(kDisplayImageSize);
    cv::imshow("Upsampled LR #1", disp_lr_1.GetVisualizationImage());

    ImageData disp_ground_truth = ground_truth;
    disp_ground_truth.ResizeImage(kDisplayImageSize);
    cv::imshow("Ground Truth", disp_ground_truth.GetVisualizationImage());

    ImageData disp_result = solver_result;
    disp_result.ResizeImage(kDisplayImageSize);
    cv::imshow("Solver Result", disp_result.GetVisualizationImage());

    cv::waitKey(0);
  }
}

// This test uses the full image formation model, including blur, and attempts
// to reconstruct the image using total variation regularization. The
// reconstructed image should not be perfect, but should be close enough.
TEST(MapSolver, RegularizationTest) {
  const cv::Mat image = cv::imread(kTestIconPath, CV_LOAD_IMAGE_COLOR);
  ImageData ground_truth(image);
  // Make sure the image size is divisible by the scale.
  ground_truth.ResizeImage(cv::Size(27, 27));
  const cv::Size image_size = ground_truth.GetImageSize();

  // Build the image model. 3x downsampling.
  const int downsampling_scale = 3;
  const super_resolution::MotionShiftSequence motion_shift_sequence({
    super_resolution::MotionShift(0, 0),
    // super_resolution::MotionShift(0, 1),
    super_resolution::MotionShift(0, 2),
    super_resolution::MotionShift(1, 0),
    // super_resolution::MotionShift(1, 1),
    super_resolution::MotionShift(1, 2),
    super_resolution::MotionShift(2, 0),
    // super_resolution::MotionShift(2, 1),
    // super_resolution::MotionShift(2, 2)
  });
  super_resolution::ImageModelParameters model_parameters;
  model_parameters.scale = downsampling_scale;
  model_parameters.motion_sequence = motion_shift_sequence;
  model_parameters.blur_radius = 3;
  model_parameters.blur_sigma = 3.0;
  const super_resolution::ImageModel image_model =
      super_resolution::ImageModel::CreateImageModel(model_parameters);

  // Additive noise (degradation image model only).
  model_parameters.noise_sigma = 10.0;
  const super_resolution::ImageModel image_model_with_noise =
      super_resolution::ImageModel::CreateImageModel(model_parameters);

  // Generate the low-res images using the image model.
  const int num_images = motion_shift_sequence.GetNumMotionShifts();
  std::vector<ImageData> low_res_images;
  for (int i = 0; i < num_images; ++i) {
    const super_resolution::ImageData low_res_image =
        image_model_with_noise.ApplyToImage(ground_truth, i);
    low_res_images.push_back(low_res_image);
  }

  // Set the initial estimate as the upsampling of the referece image, in this
  // case low_res_images[0], since it has no motion shift.
  ImageData initial_estimate = low_res_images[0];
  initial_estimate.ResizeImage(
      downsampling_scale, super_resolution::INTERPOLATE_LINEAR);

  // Create the solver and attempt to solve with TV regularization.
  super_resolution::IrlsMapSolver solver_with_tv_regularization(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  // Add regularizer.
  const std::shared_ptr<super_resolution::Regularizer> tv_regularizer(
      new super_resolution::TotalVariationRegularizer(
          image_size, ground_truth.GetNumChannels()));
  solver_with_tv_regularization.AddRegularizer(tv_regularizer, 0.01);
  // Solve.
  const ImageData solver_result_with_tv_regularization =
      solver_with_tv_regularization.Solve(initial_estimate);

  // Create a solver with BTV regularization.
  super_resolution::IrlsMapSolver solver_with_btv_regularization(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  const std::shared_ptr<super_resolution::Regularizer> btv_regularizer(
      new super_resolution::BilateralTotalVariationRegularizer(
          image_size, ground_truth.GetNumChannels(), 3, 0.5));
  solver_with_btv_regularization.AddRegularizer(btv_regularizer, 0.01);
  const ImageData solver_result_with_btv_regularization =
      solver_with_btv_regularization.Solve(initial_estimate);

  // Create the solver without regularization.
  super_resolution::IrlsMapSolver solver_unregularized(
      kDefaultSolverOptions, image_model, low_res_images, kPrintSolverOutput);
  const ImageData solver_result_unregularized =
      solver_unregularized.Solve(initial_estimate);

  // PSNR is a relative quality metric, so expect PSNR
  const super_resolution::PeakSignalToNoiseRatioEvaluator psnr_evaluator(
      ground_truth);
  const double psnr_with_tv_regularization =
      psnr_evaluator.Evaluate(solver_result_with_tv_regularization);
  const double psnr_with_btv_regularization =
      psnr_evaluator.Evaluate(solver_result_with_btv_regularization);
  const double psnr_without_regularization =
      psnr_evaluator.Evaluate(solver_result_unregularized);
  EXPECT_GT(psnr_with_tv_regularization, psnr_without_regularization);
  EXPECT_GT(psnr_with_btv_regularization, psnr_with_tv_regularization);

  if (kDisplaySolverResults) {
    ImageData disp_ground_truth = ground_truth;
    disp_ground_truth.ResizeImage(kDisplayImageSize);
    cv::imshow("Ground Truth", disp_ground_truth.GetVisualizationImage());

    ImageData disp_upsampled = initial_estimate;
    disp_upsampled.ResizeImage(kDisplayImageSize);
    cv::imshow("Upsampled", disp_upsampled.GetVisualizationImage());

    ImageData disp_result_unregularized = solver_result_unregularized;
    disp_result_unregularized.ResizeImage(kDisplayImageSize);
    cv::imshow(
        "Solver Result Not Regularized",
        disp_result_unregularized.GetVisualizationImage());

    ImageData disp_result_with_tv_regularization =
        solver_result_with_tv_regularization;
    disp_result_with_tv_regularization.ResizeImage(kDisplayImageSize);
    cv::imshow(
        "Solver Result With TV Regularization",
        disp_result_with_tv_regularization.GetVisualizationImage());

    ImageData disp_result_with_btv_regularization =
        solver_result_with_btv_regularization;
    disp_result_with_btv_regularization.ResizeImage(kDisplayImageSize);
    cv::imshow(
        "Solver Result With BTV Regularization",
        disp_result_with_btv_regularization.GetVisualizationImage());

    cv::waitKey(0);
  }
}
