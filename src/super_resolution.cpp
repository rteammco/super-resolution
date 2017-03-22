// This binary is used to run the implemented super-resolution algorithm(s) on
// a given set of images or a video. It provides an interface for the user to
// specify parameters of the algorithm without needing to code it directly.

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "evaluation/peak_signal_to_noise_ratio.h"
#include "evaluation/structural_similarity.h"
#include "hyperspectral/spectral_pca.h"
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
#include "util/data_loader.h"
#include "util/macros.h"
#include "util/util.h"
#include "wavelet/wavelet_transform.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

using super_resolution::ImageData;
using super_resolution::ImageModel;

// Input images (required):
DEFINE_string(data_path, "",
    "Path to an input file or directory to super resolve.");

// Set to true to generate the low-resolution images from an image file given
// as the data_path argument. If this is not set, data_path should be a
// directory of low-resolution images to super-resolve. Use this flag for
// testing and evaluation purposes. Number of images generated depends on the
// motion sequence.
DEFINE_bool(generate_lr_images, false,
    "Super-resolve images generated from high-res file at data_path.");
DEFINE_double(noise_sigma, 0.0,
    "Additive noise std. deviation (only if --generate_lr_images is set).");
DEFINE_int32(number_of_frames, 4,
    "The number of frames to generate (only if --generate_lr_images is set).");

// Image model parameters:
DEFINE_int32(upsampling_scale, 2,
    "The amount by which to super-resolve the image(s).");
DEFINE_int32(blur_radius, 3,
    "The radius of the blur kernel. Set to 0 to inactivate blurring.");
DEFINE_double(blur_sigma, 1.0,
    "The sigma value of the Gaussian blur. Set to 0 to inactivate blurring.");
DEFINE_string(motion_sequence_path, "",
    "Path to a file containing the motion shifts for each image.");

// Solver strategy parameters:
// TODO: Add support for different solver strategies (e.g. ADMM).
DEFINE_int32(optimization_iterations, 20,
    "Max number of optimization iterations (e.g. number of IRLS iterations).");
DEFINE_bool(solve_in_wavelet_domain, false,
    "Run super-resolution in the wavelet domain (experimental).");
DEFINE_bool(interpolate_color, false,
    "Run SR only on the luminance channel and interpolate colors later.");
DEFINE_bool(solve_in_pca_space, false,
    "Run SR on PCA space of the spectra domain (HS images only).");
DEFINE_int32(num_pca_components, 0,
    "Number of PCA components to use (0 = all) if solve_in_pca_space is set.");
DEFINE_double(pca_retained_variance, 0.0,
    "Retained variance for PCA (1.0 = all, 0.0 = use num_pca_components).");

// Regularization options:
// TODO: Add support for multiple regularizers simultaneously.
DEFINE_string(regularizer, "tv",
    "The regularizer to use ('tv', '3dtv', 'btv').");
DEFINE_int32(btv_scale_range, 3,
    "The range (window size) for BTV regularization. Minumum range is 1.");
DEFINE_double(btv_spatial_decay, 0.5,
    "The spatial decay factor for BTV regularization (0 < decay <= 1).");
DEFINE_double(regularization_parameter, 0.01,
    "The regularization parameter (lambda). 0 to not use regularization.");

// Solver parameters:
DEFINE_string(solver, "cg",
    "The least squares solver to use ('cg' or 'lbfgs').");
DEFINE_int32(solver_iterations, 50,
    "The maximum number of solver iterations.");
DEFINE_bool(use_numerical_differentiation, false,
    "Use numerical differentiation (very slow) for test purposes.");

// Evaluation and testing:
DEFINE_bool(verbose, false,
    "Solver will log progress and image stats will be printed.");
DEFINE_string(evaluators, "",
    "Comma-delimited evaluation metrics to test against (e.g. 'psnr,ssim').");

// What to do with the results (optional):
DEFINE_string(display_mode, "",
    "'result' to display; 'compare' to also display bilinear upsampling.");
DEFINE_string(result_path, "",
    "Name of file (with path) where the result image will be saved.");

// This struct is used to track input data.
struct InputData {
  ImageData high_res_image;  // Optional (if ground truth is passed in).
  std::vector<ImageData> low_res_images;  // Necessary for super-resolution.
};

// Runs the solver on the given inputs and returns the output. All solver
// options are set based on the user input flags. Post-processing the result
// (such as changing color space back to BGR) is not handled here.
ImageData SetupAndRunSolver(
    const ImageModel& image_model,
    const std::vector<ImageData>& input_images,
    const ImageData& initial_estimate) {

  // Set up the solver.
  // TODO: let the user choose the solver (once more solvers are supported).
  super_resolution::IRLSMapSolverOptions solver_options;
  if (FLAGS_solver == "cg") {
    solver_options.least_squares_solver = super_resolution::CG_SOLVER;
    LOG(INFO) << "Using conjugate gradient solver.";
  } else if (FLAGS_solver == "lbfgs") {
    solver_options.least_squares_solver = super_resolution::LBFGS_SOLVER;
    LOG(INFO) << "Using LBFGS solver.";
  } else {
    LOG(WARNING) << "Invalid solver flag. Using default (conjugate gradient).";
  }
  solver_options.max_num_irls_iterations = FLAGS_optimization_iterations;
  solver_options.max_num_solver_iterations = FLAGS_solver_iterations;
  solver_options.use_numerical_differentiation =
      FLAGS_use_numerical_differentiation;
  super_resolution::IRLSMapSolver solver(
      solver_options, image_model, input_images);
  if (!FLAGS_verbose) {
    solver.Stfu();
  }

  // Add the appropriate regularizer based on user input.
  // TODO: support for multiple regularizers at once.
  std::shared_ptr<super_resolution::Regularizer> regularizer;
  if (FLAGS_regularization_parameter > 0.0) {
    if (FLAGS_regularizer == "tv" || FLAGS_regularizer == "3dtv") {
      regularizer =
          std::shared_ptr<super_resolution::Regularizer>(
              new super_resolution::TotalVariationRegularizer(
                  initial_estimate.GetImageSize(),
                  initial_estimate.GetNumChannels()));
      if (FLAGS_regularizer == "3dtv") {
        dynamic_cast<super_resolution::TotalVariationRegularizer*>(
            regularizer.get())->SetUse3dTotalVariation(true);
      }
    } else if (FLAGS_regularizer == "btv") {
      regularizer =
          std::shared_ptr<super_resolution::Regularizer>(
              new super_resolution::BilateralTotalVariationRegularizer(
                  initial_estimate.GetImageSize(),
                  initial_estimate.GetNumChannels(),
                  FLAGS_btv_scale_range,
                  FLAGS_btv_spatial_decay));
    } else {
      LOG(WARNING) << "Unknown regularizer option '" << FLAGS_regularizer
                   << "'. Using default Total Variation regularizer.";
      FLAGS_regularizer = "tv";
      regularizer =
          std::shared_ptr<super_resolution::Regularizer>(
              new super_resolution::TotalVariationRegularizer(
                  initial_estimate.GetImageSize(),
                  initial_estimate.GetNumChannels()));
    }
    solver.AddRegularizer(regularizer, FLAGS_regularization_parameter);
    LOG(INFO) << "Added " << FLAGS_regularizer
              << " regularizer with regularization parameter "
              << FLAGS_regularization_parameter;
  }

  // Run the solver and time it.
  LOG(INFO) << "Super-resolving from " << input_images.size() << " images...";
  const auto start_time = std::chrono::steady_clock::now();
  ImageData result = solver.Solve(initial_estimate);
  const auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_time_seconds = end_time - start_time;
  LOG(INFO) << "Done! Finished in "
            << elapsed_time_seconds.count() << " seconds.";

  return result;
}

ImageData SolveInWaveletDomain(
    const ImageModel& image_model,
    const std::vector<ImageData>& input_images) {

  // Generate coefficients for each input image.
  std::vector<ImageData> input_dwt_ll_coefficients;
  std::vector<ImageData> input_dwt_lh_coefficients;
  std::vector<ImageData> input_dwt_hl_coefficients;
  std::vector<ImageData> input_dwt_hh_coefficients;
  for (const ImageData& input : input_images) {
    super_resolution::wavelet::WaveletCoefficients coefficients
        = super_resolution::wavelet::WaveletTransform(input);
    input_dwt_ll_coefficients.push_back(coefficients.ll);
    input_dwt_lh_coefficients.push_back(coefficients.lh);
    input_dwt_hl_coefficients.push_back(coefficients.hl);
    input_dwt_hh_coefficients.push_back(coefficients.hh);
  }

  // Run super-resolution on each component individually.
  // TODO: Allow selecting which of these actually get super-resolved.
  // LL:
  ImageData initial_estimate_ll = input_dwt_ll_coefficients[0];
  initial_estimate_ll.ResizeImage(
      FLAGS_upsampling_scale, super_resolution::INTERPOLATE_LINEAR);
  ImageData result_ll = SetupAndRunSolver(
      image_model, input_dwt_ll_coefficients, initial_estimate_ll);
  // LH:
  ImageData initial_estimate_lh = input_dwt_lh_coefficients[0];
  initial_estimate_lh.ResizeImage(
      FLAGS_upsampling_scale, super_resolution::INTERPOLATE_LINEAR);
  ImageData result_lh = SetupAndRunSolver(
      image_model, input_dwt_lh_coefficients, initial_estimate_lh);
  // HL:
  ImageData initial_estimate_hl = input_dwt_hl_coefficients[0];
  initial_estimate_hl.ResizeImage(
      FLAGS_upsampling_scale, super_resolution::INTERPOLATE_LINEAR);
  ImageData result_hl = SetupAndRunSolver(
      image_model, input_dwt_hl_coefficients, initial_estimate_hl);
  // HH:
  ImageData initial_estimate_hh = input_dwt_hh_coefficients[0];
  initial_estimate_hh.ResizeImage(
      FLAGS_upsampling_scale, super_resolution::INTERPOLATE_LINEAR);
  ImageData result_hh = SetupAndRunSolver(
      image_model, input_dwt_hh_coefficients, initial_estimate_hh);

  // Merge and reconstruct. Because of size precision errors where the lower
  // resolutions don't divide evenly by the upsampling scale, scale the ll
  // coefficient to the same size as the others. Then once reconstructed, scale
  // everything back to the target size. This offset should be only one pixel.
  super_resolution::wavelet::WaveletCoefficients result_coefficients;
  result_coefficients.ll = result_ll;
  result_coefficients.lh = result_lh;
  result_coefficients.hl = result_hl;
  result_coefficients.hh = result_hh;
  // result_coefficients.ll.ResizeImage(  // TODO: Put back if needed.
  //     result_coefficients.lh.GetImageSize(),
  //     super_resolution::INTERPOLATE_CUBIC);
  ImageData result = super_resolution::wavelet::InverseWaveletTransform(
      result_coefficients);

  const cv::Size original_size = input_images[0].GetImageSize();
  const cv::Size target_size(
      original_size.width * FLAGS_upsampling_scale,
      original_size.height * FLAGS_upsampling_scale);
  result.ResizeImage(target_size, super_resolution::INTERPOLATE_CUBIC);
  return result;
}

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Super resolution.");

  REQUIRE_ARG(FLAGS_data_path);

  // Create the forward image model.
  super_resolution::ImageModelParameters model_parameters;
  model_parameters.scale = FLAGS_upsampling_scale;
  model_parameters.blur_radius = FLAGS_blur_radius;
  model_parameters.blur_sigma = FLAGS_blur_sigma;
  model_parameters.motion_sequence_path = FLAGS_motion_sequence_path;

  const ImageModel image_model =
      ImageModel::CreateImageModel(model_parameters);

  // Load in or generate the low-resolution images.
  InputData input_data;
  if (FLAGS_generate_lr_images) {
    LOG(INFO) << "Generating low-resolution images from ground truth.";
    input_data.high_res_image =
        super_resolution::util::LoadImage(FLAGS_data_path);
    // Create another image model with the noise module to generate LR images.
    model_parameters.noise_sigma = FLAGS_noise_sigma;
    ImageModel image_model_with_noise =
        ImageModel::CreateImageModel(model_parameters);
    for (int i = 0; i < FLAGS_number_of_frames; ++i) {
      const ImageData low_res_frame =
          image_model_with_noise.ApplyToImage(input_data.high_res_image, i);
      input_data.low_res_images.push_back(low_res_frame);
    }
  } else {
    input_data.low_res_images =
        super_resolution::util::LoadImages(FLAGS_data_path);
  }
  CHECK_GT(input_data.low_res_images.size(), 0)
      << "At least one low-resolution image is required for super-resolution.";

  // Create an interpolated (bilinear upsampled) image as a reference.
  ImageData upsampled_image = input_data.low_res_images[0];
  upsampled_image.ResizeImage(
      FLAGS_upsampling_scale, super_resolution::INTERPOLATE_LINEAR);

  // If the interpolate_color flag is set, only run super-resolution on the
  // luminance channel and interpolate color information after. This will only
  // work on color images and will not work for grayscale or hyperspectral
  // inputs.
  if (FLAGS_interpolate_color) {
    LOG(INFO) << "Super-resolving only the luminance channel.";
    for (int i = 0; i < input_data.low_res_images.size(); ++i) {
      // TODO: support for more color spaces.
      // Switch color mode (true = use only the luminance channel for SR).
      input_data.low_res_images[i].ChangeColorSpace(
          super_resolution::SPECTRAL_MODE_COLOR_YCRCB, true);
    }
  }

  // If solve_in_pca_space flag was set, convert the image(s) to the spectral
  // PCA domain and solve. Then convert them back after the solver finishes.
  // Cannot use this option if using the color interpolation scheme.
  std::unique_ptr<super_resolution::SpectralPCA> spectral_pca;
  if (FLAGS_solve_in_pca_space && !FLAGS_interpolate_color) {
    // TODO: Get the sampling options and number of PCA bands from user args!
    if (FLAGS_pca_retained_variance > 0.0) {
      spectral_pca = std::unique_ptr<super_resolution::SpectralPCA>(
          new super_resolution::SpectralPCA(
              input_data.low_res_images, FLAGS_pca_retained_variance));
    } else {
      spectral_pca = std::unique_ptr<super_resolution::SpectralPCA>(
          new super_resolution::SpectralPCA(
              input_data.low_res_images, FLAGS_num_pca_components));
    }
    for (int i = 0; i < input_data.low_res_images.size(); ++i) {
      input_data.low_res_images[i] =
          spectral_pca->GetPCAImage(input_data.low_res_images[i]);
    }
    LOG(INFO) << "Super-resolving in PCA space with "
              << input_data.low_res_images[0].GetNumChannels()
              << " PCA components.";
  }

  // Use an interpolated (bilinear upsampled) image the initial estimate. This
  // is done after any other conversions to keep it in the same spectral space
  // that the solver will operate in.
  ImageData initial_estimate = input_data.low_res_images[0];
  initial_estimate.ResizeImage(
      FLAGS_upsampling_scale, super_resolution::INTERPOLATE_LINEAR);

  // Run super-resolution in the selected domain.
  ImageData result;
  if (FLAGS_solve_in_wavelet_domain) {
    result = SolveInWaveletDomain(image_model, input_data.low_res_images);
  } else {
    // Solving is handled in the SetupAndRunSolver function above.
    result = SetupAndRunSolver(
        image_model, input_data.low_res_images, initial_estimate);
  }

  // If SR was only done on the luminance channel, interpolate the colors now
  // and change the color space back to BGR.
  //
  // Note that the colors we're interpolating are in the luminance-dominant
  // color space (not BGR), and so we must interpolate the initial estimate,
  // which is in the same color space as the solved image, rather than the
  // reference upsampled image which was never converted from BGR.
  if (FLAGS_interpolate_color) {
    result.InterpolateColorFrom(initial_estimate);
    result.ChangeColorSpace(super_resolution::SPECTRAL_MODE_COLOR_BGR);
  }

  // If SR was done in PCA space, convert back to regular space.
  if (FLAGS_solve_in_pca_space && !FLAGS_interpolate_color) {
    result = spectral_pca->ReconstructImage(result);
  }

  // If an evaluation criteria is passed in and the high-resolution image is
  // available, display the evaluation results.
  if (FLAGS_generate_lr_images && !FLAGS_evaluators.empty()) {
    std::vector<std::string> evaluators =
        super_resolution::util::SplitString(FLAGS_evaluators, ',');
    for (const std::string& evaluator_arg : evaluators) {
      const std::string evaluator =
          super_resolution::util::TrimString(evaluator_arg);
      if (evaluator == "psnr") {
        super_resolution::PeakSignalToNoiseRatioEvaluator psnr_evaluator(
            input_data.high_res_image);
        const double upsampled_psnr = psnr_evaluator.Evaluate(upsampled_image);
        const double result_psnr = psnr_evaluator.Evaluate(result);
        LOG(INFO) << "PSNR score on upsampled: " << upsampled_psnr;
        LOG(INFO) << "PSNR score on result:    " << result_psnr;
      } else if (evaluator == "ssim") {
        super_resolution::StructuralSimilarityEvaluator ssim_evaluator(
            input_data.high_res_image);
        const double upsampled_ssim = ssim_evaluator.Evaluate(upsampled_image);
        const double result_ssim = ssim_evaluator.Evaluate(result);
        LOG(INFO) << "SSIM score on upsampled: " << upsampled_ssim;
        LOG(INFO) << "SSIM score on result:    " << result_ssim;
      } else {
        LOG(ERROR) << "Unknown/unsupported evaluator '" << evaluator << "'.";
      }
    }
  }
  if (FLAGS_verbose) {
    result.GetImageDataReport().Print();
  }

  if (FLAGS_display_mode == "result") {
    super_resolution::util::DisplayImage(result, "Result");
  } else if (FLAGS_display_mode == "compare") {
    std::vector<ImageData> display_images = {result, upsampled_image};
    std::string display_title = "Super-Resolution vs. Linear Interpolation";
    if (FLAGS_generate_lr_images) {
      display_images.insert(display_images.begin(), input_data.high_res_image);
      display_title = "Ground Truth vs. " + display_title;
    }
    super_resolution::util::DisplayImagesSideBySide(
        display_images, display_title);
  }

  return EXIT_SUCCESS;
}
