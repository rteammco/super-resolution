#include "solvers/tv_regularizer.h"

#include <cmath>
#include <vector>

#include "glog/logging.h"

namespace super_resolution {

std::vector<double> TotalVariationRegularizer::ComputeResiduals(
    const double* image_data) const {

  CHECK_NOTNULL(image_data);

  std::vector<double> residuals(image_size_.width * image_size_.height);
  for (int row = 0; row < image_size_.height; ++row) {
    for (int col = 0; col < image_size_.width; ++col) {
      const int index = row * image_size_.width + col;
      double total_variation = 0;
      // Compute y gradient (between this pixel and the one below it).
      if (row + 1 < image_size_.height) {
        const int y_neighbor_index = (row + 1) * image_size_.width + col;
        const double y_variation =
            image_data[y_neighbor_index] - image_data[index];
        total_variation += y_variation * y_variation;
      }
      // Compute x gradient (between this pixel and the one to the right).
      if (col + 1 < image_size_.width) {
        const int x_neighbor_index = row * image_size_.width + (col + 1);
        const double x_variation =
            image_data[x_neighbor_index] - image_data[index];
        total_variation += x_variation * x_variation;
      }
      residuals[index] = sqrt(total_variation);
    }
  }
  return residuals;
}

}  // namespace super_resolution
