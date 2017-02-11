// The ImageModel class specifies a forward image formation model that can
// produce a LR image given an HR image. It is defined as a series of
// degradations (DegradationOperators) that downsample, blur, move, etc. the
// original HR image to simulate an observation.
//
// The model can be used to either generate simulated image data, or as a
// component in some objective functions, such as that used in MAP.

#ifndef SRC_IMAGE_MODEL_IMAGE_MODEL_H_
#define SRC_IMAGE_MODEL_IMAGE_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "image/image_data.h"
#include "image_model/degradation_operator.h"
#include "motion/motion_shift.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

// A set of parameters for generating a standard ImageModel using the static
// CreateImageModel() method.
struct ImageModelParameters {
  // Downsampling (D).
  int scale = 2;

  // Blur (B). Set either values to 0 to not include blur.
  int blur_radius = 3;
  double blur_sigma = 1.0;

  // Motion (M). Set the file path of a motion sequence path to load it from a
  // file, or set the motion shift sequence. Either can be used to make a
  // motion operator.
  std::string motion_sequence_path = "";
  MotionShiftSequence motion_sequence;

  // Noise. Set to a positive value to include noise. This is just for
  // generating artificial data. Do not add noise for modeling a forward image
  // model in super-resolution.
  double noise_sigma = 0.0;
};

class ImageModel {
 public:
  // A generator for the standard image model. Define the ImageModelParameters
  // for adding the appropriate degradation operators.
  static ImageModel CreateImageModel(const ImageModelParameters& parameters);

  // ImageModel keeps track of downsampling factor (e.g. downsampling_scale = 2
  // for 2x super-resolution increase). The downsampling operator is NOT
  // created here and must be added manually with AddDegradationOperator().
  explicit ImageModel(const int downsampling_scale);

  // Adds a DegradationOperator to the model. These operators should be added in
  // the order that they need to be applied. For example, if the image model is
  //    y = DBMx + n
  // where y and x are the LR and HR images, respectively, then add order
  // should be M, B, D, n.
  // Note that additive operators come last due to the order of operations.
  //
  // Because the DegradationOperator is pure virtual, add the operator using
  // std::move. For example,
  //   std::shared_ptr<DownsamplingModule> downsampling_module(
  //       new DownsamplingModule(parameters.scale));
  //   image_model.AddDegradationOperator(downsampling_module);
  void AddDegradationOperator(
      const std::shared_ptr<DegradationOperator> degradation_operator);

  // Apply this forward model to the given image at the given index in the
  // multiframe sequence. The degraded image is returned as a new image, with
  // the original ImageData being unaffected.
  ImageData ApplyToImage(const ImageData& image_data, const int index) const;

  // Same as the above ApplyToImage, but this version modifies the given
  // ImageData instead of returning a modified copy.
  void ApplyToImage(ImageData* image_data, const int index) const;

  // Applies the transpose of the operators. For example, if the image model is
  // defined on as DBM, then the transpose is defined as M'B'D'. Operator
  // transpose implementations must be defined in every DegradationOperator.
  void ApplyTransposeToImage(ImageData* image_data, const int index) const;

  // NOTE: This function is very slow, and its only purpose is to test solver
  // implementations on very small data sets. Some operators may not support
  // parameters exceeding a certain matrix size.
  //
  // Returns the combined matrix representation of the image model by
  // multiplying all of the operator matrices. These operators will be
  // multiplied in reverse order so that they are applied to the vector image
  // in the same order that they were added. For example, if the model is
  //    y = DBMx
  // and the operators were added in order M, B, D, then the matrix returned
  // will be A = DBM, first multiplying B*M, and then that result by D.
  //
  // "image_size" is the size of the image to be multiplied. This must be
  // specified correctly to build the degradation matrices. "index" is the
  // index of the LR image for which this operator will be generated.
  //
  // At least one DegradationOperator must be available, otherwise this will
  // cause a check fail.
  cv::Mat GetModelMatrix(
      const cv::Size& image_size, const int index) const;

  // Returns the downsampling scale.
  int GetDownsamplingScale() const {
    return downsampling_scale_;
  }

 private:
  // An ordered list of degradation operators, to be applied in this order. We
  // keep pointers because the DegradationOperator class is pure virtual.
  std::vector<std::shared_ptr<DegradationOperator>> degradation_operators_;

  // The ImageModel keeps track of the downsampling scale factor.
  const int downsampling_scale_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_IMAGE_MODEL_H_
