// This binary is used to generate low-resolution images from a given
// high-resolution ground truth image. Use this to generate data before running
// the SuperResolution binary for algorithm testing and evaluation.
//
// TODO: This currently only supports RGB or Grayscale images. Extend this to
// support hyperspectral data as well.

#include <string>
#include <vector>

#include "image/image_data.h"
#include "image_model/additive_noise_module.h"
#include "image_model/blur_module.h"
#include "image_model/downsampling_module.h"
#include "image_model/image_model.h"
#include "image_model/motion_module.h"
#include "motion/motion_shift.h"
#include "util/data_loader.h"
#include "util/macros.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

using super_resolution::ImageData;

// Required input and output files.
DEFINE_string(input_image, "",
    "Path to the HR image that will be used to generate the LR images.");
DEFINE_string(output_image_dir, "",
    "Path to a directory that will contain all of the generated LR images.");

// Motion estimate file I/O parameters.
DEFINE_string(motion_sequence_path, "",
    "Path to a text file containing a simulated motion sequence.");

// Parameters for the low-resolution image generation.
DEFINE_int32(blur_radius, 0,
    "The radius of the Gaussian blur kernel. If 0, no blur will be added.");
DEFINE_double(blur_sigma, 0.0,
    "The sigma of the Gaussian blur kernel. If 0, no blur will be added.");
DEFINE_double(noise_sigma, 0.0,
    "Standard deviation of the additive noise. If 0, no noise will be added.");
DEFINE_int32(downsampling_scale, 2,
    "The scale by which the HR image will be downsampled.");
DEFINE_int32(number_of_frames, 4,
    "The number of LR images that will be generated.");

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv,
      "Generate low-resolution frames from a high-resolution image.");

  REQUIRE_ARG(FLAGS_input_image);
  REQUIRE_ARG(FLAGS_output_image_dir);

  const cv::Mat image = cv::imread(FLAGS_input_image, cv::IMREAD_UNCHANGED);
  const ImageData image_data(image);

  // Set up the ImageModel with all the parameters specified by the user. This
  // model will be used to generate the degradated images.
  super_resolution::ImageModelParameters model_parameters;
  model_parameters.scale = FLAGS_downsampling_scale;
  model_parameters.blur_radius = FLAGS_blur_radius;
  model_parameters.blur_sigma = FLAGS_blur_sigma;
  model_parameters.motion_sequence_path = FLAGS_motion_sequence_path;
  model_parameters.noise_sigma = FLAGS_noise_sigma;

  super_resolution::ImageModel image_model =
      super_resolution::ImageModel::CreateImageModel(model_parameters);

  for (int i = 0; i < FLAGS_number_of_frames; ++i) {
    const ImageData low_res_frame = image_model.ApplyToImage(image_data, i);
    // Write the file.
    std::string image_path =
        FLAGS_output_image_dir + "/low_res_" + std::to_string(i) + ".jpg";
    super_resolution::util::SaveImage(low_res_frame, image_path);
    LOG(INFO) << "Generated output image " << image_path;
  }

  return EXIT_SUCCESS;
}
