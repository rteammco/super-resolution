#include "evaluation/structural_similarity.h"

#include "glog/logging.h"

namespace super_resolution {
namespace {

// Computes the average pixel intensity of an image.
double ComputeAveragePixelIntensity(const ImageData& image) {
  const int num_channels = image.GetNumChannels();
  const int num_pixels = image.GetNumPixels();
  double intensity_sum = 0.0;
  for (int channel = 0; channel < num_channels; ++channel) {
    for (int pixel = 0; pixel < num_pixels; ++pixel) {
      intensity_sum += image.GetPixelValue(channel, pixel);
    }
  }
  return intensity_sum / static_cast<double>(num_channels * num_pixels);
}

// Returns the covariance between the pixel intensities of the two given
// images. The mean values of those images are required as well.
double ComputePixelIntensityCovariance(
    const ImageData& image1,
    const double mean1,
    const ImageData& image2,
    const double mean2) {

  const int num_channels = image1.GetNumChannels();
  const int num_pixels = image1.GetNumPixels();
  double covariance = 0.0;
  for (int channel = 0; channel < num_channels; ++channel) {
    for (int pixel = 0; pixel < num_pixels; ++pixel) {
      const double diff1 = image1.GetPixelValue(channel, pixel) - mean1;
      const double diff2 = image2.GetPixelValue(channel, pixel) - mean2;
      covariance += diff1 * diff2;
    }
  }
  return covariance / static_cast<double>(num_channels * num_pixels);
}

// Computes the variance in pixel intensities of a single image.
double ComputePixelIntensityVariance(
    const ImageData& image, const double mean) {

  // The variance is just the covariance of the image with itself.
  return ComputePixelIntensityCovariance(image, mean, image, mean);
}

}  // namespace

// Constructor: pre-compute values for ground truth image.
StructuralSimilarityEvaluator::StructuralSimilarityEvaluator(
    const ImageData& ground_truth,
    const double k1,
    const double k2,
    const double image_scale)
    : GroundTruthEvaluator(ground_truth) {

  ground_truth_mean_ = ComputeAveragePixelIntensity(ground_truth);
  ground_truth_variance_ =
      ComputePixelIntensityVariance(ground_truth, ground_truth_mean_);
  c1_ = k1 * image_scale;
  c1_ = c1_ * c1_;
  c2_ = k2 * image_scale;
  c2_ = c2_ * c2_;
}

double StructuralSimilarityEvaluator::Evaluate(const ImageData& image) const {
  CHECK_EQ(image.GetNumChannels(), ground_truth_.GetNumChannels());
  ImageData evaluation_image = image;
  if (image.GetImageSize() != ground_truth_.GetImageSize()) {
    LOG(WARNING) << "Image size is different from ground truth: "
                 << image.GetImageSize() << " vs. "
                 << ground_truth_.GetImageSize() << ". "
                 << "Resizing image to run evaluation.";
    evaluation_image.ResizeImage(image.GetImageSize(), INTERPOLATE_LINEAR);
  }

  const double image_mean = ComputeAveragePixelIntensity(evaluation_image);
  const double image_variance =
      ComputePixelIntensityVariance(evaluation_image, image_mean);
  const double covariance = ComputePixelIntensityCovariance(
      evaluation_image, image_mean, ground_truth_, ground_truth_mean_);

  const double numerator_1 = 2 * ground_truth_mean_ * image_mean + c1_;
  const double numerator_2 = 2 * covariance + c2_;
  const double denominator_1 =
      ground_truth_mean_ * ground_truth_mean_ +
      image_mean * image_mean +
      c1_;
  const double denominator_2 = ground_truth_variance_ + image_variance + c2_;

  return (numerator_1 * numerator_2) / (denominator_1 * denominator_2);
}

}  // namespace super_resolution
