// A GroundTruthEvaluator compares a given image to the ground truth. The value
// returned is based on the quality of the given image as determined by the
// implemented evaluation metric.

#ifndef SRC_EVALUATION_GROUND_TRUTH_EVALUATOR_H_
#define SRC_EVALUATION_GROUND_TRUTH_EVALUATOR_H_

#include "image/image_data.h"

// Abstract class.
class GroundTruthEvaluator {
 public:
  explicit GroundTruthEvaluator(const ImageData& ground_truth)
      : ground_truth_(ground_truth) {}

  double Evaluate(const ImageData& image) const = 0;

 private:
  const ImageData& ground_truth_;
};

#endif  // SRC_EVALUATION_GROUND_TRUTH_EVALUATOR_H_
