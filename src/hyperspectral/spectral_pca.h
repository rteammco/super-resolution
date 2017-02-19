// This object uses the given set of images (one or more required) to find the
// PCA basis for the spectral bands. The PCA basis consists of eigenvalues and
// eigenvectors which are then used to convert images to and from the PCA
// space. Note that a hyperspectral image is required to get a meaningful
// decomposition.
//
// PCA on the hyperspectral bands is computed as follows:
//   x_i = ith pixel vector (b x 1 where b is the number of spectral bands).
//   mx = mean(x_i)                           <= mean pixel vector
//   Cx = (1/m) sum((x_i - mx)*(x_i - mx)^T)  <= spectral covariance matrix
//   PCA(Cx) = ADA^T
//     ^ D is the diagonal eigenvalue matrix.
//     ^ Columns of A are the eigenvectors.
//
// Now, given a pixel vector x, the corresponding PCA pixel vector y = A^T*x.
// Reconstruct x from y as x = A*y.
//
// Further, an approximation can be estimated. Sort the eigenvalues in D in
// descending order (d_1 >= d_2 >= ... >= d_b) and the corresponding
// eigenvectors in A accordingly. Then let B be a subset of the eigenvectors of
// A corresponding to the top k eigenvalues (d_1, d_2, ..., d_k), k << b. Then
// an approximate PCA representation of a given pixel vector x can be given as
// y = B^T*x, and an approximate reconstruction would be x ~= B*y.
//
// The approximation can reduce the dimensionality of the search space by
// exploiting correlations between the spectra. The sum of the first k (sorted)
// eigenvalues reflects the amount of information preserved. The top few
// eigenvalues typically contain almost all of the information (e.g. 99%).
//
// This object does everything automatically. Use as follows:
//   SpectralPca spectral_pca(low_res_images);
//   pca_images[0] = spectral_pca.GetPcaImage(low_res_images[0]);
//   hr_estimate = spectral_pca.ReconstructImage(high_res_pca_estimate);

#ifndef SRC_HYPERSPECTRAL_SPECTRAL_PCA_H_
#define SRC_HYPERSPECTRAL_SPECTRAL_PCA_H_

#include <vector>

#include "image/image_data.h"

namespace super_resolution {

class SpectralPca {
 public:
  // Uses the given set of images to generate the PCA decomposition and finds
  // the top PCA bands.
  explicit SpectralPca(const std::vector<ImageData>& hyperspectral_images);

  // Returns an image with PCA spectral channels (each pixel is converted into
  // the precomputed PCA space). If num_pca_bands is positive, then the
  // returned image will have that many PCA bands (capped by the total number
  // of spectral bands available). The reconstructed image will be an
  // approximation if the number of PCA bands is less than the total number of
  // spectral bands (see the description above).
  ImageData GetPcaImage(
      const ImageData& image_data, const int num_pca_bands = 0) const;

  // Same as the above GetPcaImage(), but the given approximation amount (where
  // 0 < approximation <= 1) will be used to find the top k eigenvalues such
  // that v_1 + v_2 + ... + v_k >= approximation. This will use the minimum
  // number of eigenvectors for the PCA basis such that the approximation
  // preserves that much of the original data (see description above). The
  // returned image will then be GetPcaImage(image_data, k).
  ImageData GetPcaImage(
      const ImageData& image_data, const double approximation) const;

  // Reconstructs the original hyperspectral image by inverting the PCA
  // operation. It is assumed that the given image was obtained from the same
  // SpectralPca object using GetPcaImage() for a valid reconstruction. The
  // number of spectral bands in the given image will indicate the PCA
  // approximation used, and the reconstruction will be computed accordingly.
  ImageData ReconstructImage(const ImageData& pca_image_data) const;

 private:
  // TODO:
  // Matrix A
  // Eigenvalues D
};

}  // namespace super_resolution

#endif  // SRC_HYPERSPECTRAL_SPECTRAL_PCA_H_
