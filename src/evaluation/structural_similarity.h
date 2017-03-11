// Structural similarity (SSIM) is a metric that measures the similarity
// between two images. It's designed to correlate more with human quality
// perception than PSNR and other prior methods.

#ifndef SRC_EVALUATION_STRUCTURAL_SIMILARITY_H_
#define SRC_EVALUATION_STRUCTURAL_SIMILARITY_H_

#include "evaluation/ground_truth_evaluator.h"
#include "image/image_data.h"

namespace super_resolution {

class StructuralSimilarityEvaluator : public GroundTruthEvaluator {
 public:
  // Precomputes c1_ and c2_ as follows:
  //   c1_ = (image_scale * k1)^2
  //   c2_ = (image_scale * k2)^2
  // k1 and k2 are normalization constants, and image_scale refers to the
  // radiometric resolution of the image (e.g. 256 if an 8-bit image or 1.0 for
  // a normalized image).
  StructuralSimilarityEvaluator(
      const ImageData& ground_truth,
      const double k1 = 0.01,
      const double k2 = 0.03,
      const double image_scale = 1.0);

  // Computes the structural similarity index (SSIM) of the given image
  // compared to the ground truth reference image.
  //
  // The SSIM for a single image channel is computed as follows:
  // SSIM = (2 * u1 * u2 + c1_) * (2 * s12 + c2_) /
  //        (u1^2 * u2^2 + c1_) * (s1 + s2 + c2_)
  // where
  //    u1 = mean intensity of the ground truth image,
  //    u2 = mean intensity of the given image,
  //    s1 = variance of the ground truth image,
  //    s2 = variance of the given image,
  //    s12 = covariance between the ground truth image and the given image,
  //    c1_ and c2_ are computed as described above (see constructor).
  //
  // TODO: The total SSIM is the average SSIM across all image channels.
  // TODO: SSIM should be computed patch-wise (e.g. on 8x8 patches).
  virtual double Evaluate(const ImageData& image) const;

 private:
  // Precomputed variance and mean for the ground truth image.
  double ground_truth_mean_;
  double ground_truth_variance_;

  // These are normalization parameters for SSIM. These are based on the given
  // k1, k2, and image_scale values. The default values should be fine for
  // evaluating normalized images within the [0, 1] scale range.
  double c1_;
  double c2_;
};

}  // namespace super_resolution

#endif  // SRC_EVALUATION_STRUCTURAL_SIMILARITY_H_
