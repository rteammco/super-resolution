#ifndef SRC_WAVELET_WAVELET_TRANSFORM_H_
#define SRC_WAVELET_WAVELET_TRANSFORM_H_

#include "image/image_data.h"

namespace super_resolution {
namespace wavelet {

// Contains the four wavelet coefficients (LL, LH, HL, HH) which contain the
// low-frequency and high-frequency coefficients of the DWT image
// decomposition.
struct WaveletCoefficients {
  ImageData ll;
  ImageData lh;
  ImageData hl;
  ImageData hh;
};

// Computes a discrete wavelet transform (DWT) of the given image.
WaveletCoefficients WaveletTransform(const ImageData& image);

// Returns an image reconstructed from the given wavelet components. If the
// given coefficients were aquicred from the WaveletTransform() function using
// the same filter, the returned image should be identical to the original
// image, save for small numerical errors.
ImageData InverseWaveletTransform(const WaveletCoefficients& coefficients);

}  // namespace wavelet
}  // namespace super_resolution

#endif  // SRC_WAVELET_WAVELET_TRANSFORM_H_
