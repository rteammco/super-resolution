// This binary is used to run the implemented super-resolution algorithm(s) on
// a given set of images or a video. It provides an interface for the user to
// specify parameters of the algorithm without needing to code it directly.

#include <iostream>
#include <memory>
#include <vector>

#include "evaluation/peak_signal_to_noise_ratio.h"
#include "image/image_data.h"
#include "image_model/additive_noise_module.h"
#include "image_model/blur_module.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "optimization/irls_map_solver.h"
#include "optimization/tv_regularizer.h"
#include "util/data_loader.h"
#include "util/macros.h"
#include "util/util.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

using super_resolution::ImageData;

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

// Solver parameters:
// TODO: add support for these regularizers.
// TODO: add support for multiple regularizers simultaneously.
DEFINE_string(regularizer, "tv",
    "The regularizer to use ('tv', '3dtv', 'btv').");
DEFINE_double(regularization_parameter, 0.01,
    "The regularization parameter (lambda). 0 to not use regularization.");
DEFINE_int32(solver_iterations, 50,
    "The maximum number of solver iterations.");
DEFINE_bool(use_numerical_differentiation, false,
    "Use numerical differentiation (very slow) for test purposes.");

// Evaluation and testing:
DEFINE_bool(verbose, false,
    "Setting this will cause the solver to log progress.");
DEFINE_string(evaluator, "",  // TODO: add support for more evaluators.
    "Optionally print an evaluation metric value ('psnr').");

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

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Super resolution.");

  REQUIRE_ARG(FLAGS_data_path);

  // Create the forward image model.
  super_resolution::ImageModelParameters model_parameters;
  model_parameters.scale = FLAGS_upsampling_scale;
  model_parameters.blur_radius = FLAGS_blur_radius;
  model_parameters.blur_sigma = FLAGS_blur_sigma;
  model_parameters.motion_sequence_path = FLAGS_motion_sequence_path;

  const super_resolution::ImageModel image_model =
      super_resolution::ImageModel::CreateImageModel(model_parameters);

  // Load in or generate the low-resolution images.
  InputData input_data;
  if (FLAGS_generate_lr_images) {
    LOG(INFO) << "Generating low-resolution images from ground truth.";
    input_data.high_res_image =
        super_resolution::util::LoadImage(FLAGS_data_path);
    super_resolution::ImageModel image_model_with_noise = image_model;
    std::shared_ptr<super_resolution::AdditiveNoiseModule> noise_module;
    if (FLAGS_noise_sigma > 0.0) {
      noise_module = std::shared_ptr<super_resolution::AdditiveNoiseModule>(
          new super_resolution::AdditiveNoiseModule(FLAGS_noise_sigma));
      image_model_with_noise.AddDegradationOperator(noise_module);
    }
    for (int i = 0; i < FLAGS_number_of_frames; ++i) {
      const ImageData low_res_frame =
          image_model_with_noise.ApplyToImage(input_data.high_res_image, i);
      input_data.low_res_images.push_back(low_res_frame);
    }
  } else {
    input_data.low_res_images =
        super_resolution::util::LoadImages(FLAGS_data_path);
  }

  // Set initial estimate.
  CHECK_GT(input_data.low_res_images.size(), 0)
      << "At least one low-resolution image is required for super-resolution.";
  ImageData initial_estimate = input_data.low_res_images[0];
  initial_estimate.ResizeImage(FLAGS_upsampling_scale, cv::INTER_LINEAR);

  // Set up the solver.
  // TODO: let the user choose the solver (once more solvers are supported).
  super_resolution::MapSolverOptions solver_options;
  solver_options.max_num_solver_iterations = FLAGS_solver_iterations;
  solver_options.use_numerical_differentiation =
      FLAGS_use_numerical_differentiation;
  super_resolution::IrlsMapSolver solver(
      solver_options, image_model, input_data.low_res_images);
  if (!FLAGS_verbose) {
    solver.Stfu();
  }

  // Add the appropriate regularizer based on user input.
  // TODO: support for multiple regularizers at once.
  std::unique_ptr<super_resolution::Regularizer> regularizer;
  if (FLAGS_regularization_parameter > 0.0) {
    if (FLAGS_regularizer == "tv" || FLAGS_regularizer == "3dtv") {
      regularizer =
          std::unique_ptr<super_resolution::TotalVariationRegularizer>(
              new super_resolution::TotalVariationRegularizer(
                  initial_estimate.GetImageSize(),
                  initial_estimate.GetNumChannels()));
      if (FLAGS_regularizer == "3dtv") {
        reinterpret_cast<super_resolution::TotalVariationRegularizer*>(
            regularizer.get())->SetUse3dTotalVariation(true);
      }
    }
    solver.AddRegularizer(*regularizer, FLAGS_regularization_parameter);
    LOG(INFO) << "Added " << FLAGS_regularizer
              << " regularizer with regularization parameter "
              << FLAGS_regularization_parameter;
  }

  std::cout << "Super-resolving from "
            << input_data.low_res_images.size() << " images..." << std::endl;
  const ImageData result = solver.Solve(initial_estimate);
  std::cout << "Done!" << std::endl;

  // If an evaluation criteria is passed in and the high-resolution image is
  // available, display the evaluation results.
  if (FLAGS_generate_lr_images && !FLAGS_evaluator.empty()) {
    if (FLAGS_evaluator == "psnr") {
      super_resolution::PeakSignalToNoiseRatioEvaluator psnr_evaluator(
          input_data.high_res_image);
      const double upsampled_psnr = psnr_evaluator.Evaluate(initial_estimate);
      const double result_psnr = psnr_evaluator.Evaluate(result);
      std::cout << "PSNR score on upsampled: " << upsampled_psnr << std::endl;
      std::cout << "PSNR score on result:    " << result_psnr << std::endl;
    }
  }
  result.GetImageDataReport().Print();  // TODO

  if (FLAGS_display_mode == "result") {
    super_resolution::util::DisplayImage(result, "Result");
  } else if (FLAGS_display_mode == "compare") {
    super_resolution::util::DisplayImagesSideBySide(
        {result, initial_estimate},
        "Super-Resolution vs. Linear Interpolation");
  }

  return EXIT_SUCCESS;
}
