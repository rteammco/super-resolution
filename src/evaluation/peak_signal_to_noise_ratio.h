// The Peak Signal To Noise Ratio (PSNR) evaluation metric is one of the most
// commonly used image quality assessment algorithms. PSNR compares a ground
// truth image against an enhanced image to measure the ratio between the
// signal (i.e. the expected image intensity values) vs. the noise (the
// super-resolution algorithm errors). PSNR approximately quantifies human
// quality perception of reconstructed images.
//
// Details about PSNR can be found here:
//     https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

#ifndef SRC_EVALUATION_PEAK_SIGNAL_TO_NOISE_RATIO_H_
#define SRC_EVALUATION_PEAK_SIGNAL_TO_NOISE_RATIO_H_

#include "evaluation/ground_truth_evaluator.h"
#include "image/image_data.h"

namespace super_resolution {

class PeakSignalToNoiseRatioEvaluator : public GroundTruthEvaluator {
 public:
  using GroundTruthEvaluator::GroundTruthEvaluator;

  // Returns the PSNR value:
  //   MSE = (i/N)*sum_i(|G_i - I_i|^2)
  //       where G_i is the ith ground truth pixel, I_i is the ith pixel in the
  //       given image, and N is the total number of pixels being compared,
  //   PSNR = 10 * log_10(MAX^2 / MSE)
  //       where MAX is the maximum possible pixel intensity value (e.g. 255
  //       for 8-bit images or 1.0 in a normalized image).
  virtual double Evaluate(const ImageData& image) const;
};

}  // namespace super_resolution

#endif  // SRC_EVALUATION_PEAK_SIGNAL_TO_NOISE_RATIO_H_
