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
  StructuralSimilarityEvaluator(
      const ImageData& ground_truth,
      const double k1 = 0.01,
      const double k2 = 0.03,
      const double image_scale = 1.0);

  // TODO: Comment and implement.
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
