// This binary is used to run the implemented super-resolution algorithm(s) on
// a given set of images or a video. It provides an interface for the user to
// specify parameters of the algorithm without needing to code it directly.

#include <memory>
#include <vector>

#include "image/image_data.h"
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

// Image model parameters:
DEFINE_int32(upsampling_scale, 2,
    "The amount by which to super-resolve the image(s).");
DEFINE_int32(blur_radius, 3,
    "The radius of the blur kernel. Set to 0 to inactivate blurring.");
DEFINE_double(blur_sigma, 1.0,
    "The sigma value of the Gaussian blur. Set to 0 to inactivate blurring.");
DEFINE_string(motion_sequence_path, "",
    "Path to a file containing the motion shifts for each image.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv, "Super resolution.");

  REQUIRE_ARG(FLAGS_data_path);

  std::vector<ImageData> images = super_resolution::util::LoadImages(
      FLAGS_data_path);

  // Create the forward image model.
  super_resolution::ImageModel image_model(FLAGS_upsampling_scale);

  // Add motion module if user specified motion shift sequence. We use a
  // pointer to avoid scoping issues of creating the module in the if block.
  std::unique_ptr<super_resolution::MotionModule> motion_module;
  if (!FLAGS_motion_sequence_path.empty()) {
    super_resolution::MotionShiftSequence motion_shift_sequence;
    motion_shift_sequence.LoadSequenceFromFile(FLAGS_motion_sequence_path);
    motion_module = std::unique_ptr<super_resolution::MotionModule>(
        new super_resolution::MotionModule(motion_shift_sequence));
    image_model.AddDegradationOperator(*motion_module);
  }

  // Add a blur module if user-specified blurring values are positive.
  std::unique_ptr<super_resolution::BlurModule> blur_module;
  if (FLAGS_blur_radius > 0 && FLAGS_blur_sigma > 0) {
    blur_module = std::unique_ptr<super_resolution::BlurModule>(
        new super_resolution::BlurModule(FLAGS_blur_radius, FLAGS_blur_sigma));
    image_model.AddDegradationOperator(*blur_module);
  }

  // Add downsampling.
  const super_resolution::DownsamplingModule downsampling_module(
      FLAGS_upsampling_scale);
  image_model.AddDegradationOperator(downsampling_module);

  // Set initial estimate.
  ImageData initial_estimate = images[0];
  initial_estimate.ResizeImage(FLAGS_upsampling_scale);

  // Set up the solver with TV regularization.
  super_resolution::MapSolverOptions solver_options;
  super_resolution::IrlsMapSolver solver(solver_options, image_model, images);
  const super_resolution::TotalVariationRegularizer tv_regularizer(
      initial_estimate.GetImageSize(), initial_estimate.GetNumChannels());
  solver.AddRegularizer(tv_regularizer, 0.01);

  const ImageData result = solver.Solve(initial_estimate);
  super_resolution::util::DisplayImage(result, "Result");

  return EXIT_SUCCESS;
}
