// A GroundTruthEvaluator compares a given image to the ground truth. The value
// returned is based on the quality of the given image as determined by the
// implemented evaluation metric.

#ifndef SRC_EVALUATION_GROUND_TRUTH_EVALUATOR_H_
#define SRC_EVALUATION_GROUND_TRUTH_EVALUATOR_H_

#include "image/image_data.h"

namespace super_resolution {

// Abstract class.
class GroundTruthEvaluator {
 public:
  explicit GroundTruthEvaluator(const ImageData& ground_truth)
      : ground_truth_(ground_truth) {}

  // Returns a value to indicate the quality of the given image using the
  // implemented evaluation metric. This value will differ between
  // implementations, so see individual derived GroundTruthEvaluator objects.
  virtual double Evaluate(const ImageData& image) const = 0;

 protected:
  const ImageData& ground_truth_;
};

}  // namespace super_resolution

#endif  // SRC_EVALUATION_GROUND_TRUTH_EVALUATOR_H_
