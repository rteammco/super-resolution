#include "image_model/degradation_operator.h"

#include "util/util.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {

cv::Mat DegradationOperator::ConvertKernelToOperatorMatrix(
    const cv::Mat& kernel, const cv::Size& image_size) {

  cv::Mat operator_matrix;
  // TODO: implement this algorithm:
  /*
print 'Kernel:'
print kernel
# Image dimensions:
nrows = image.shape[0]
ncols = image.shape[1]
# Kernel dimensions:
krows = kernel.shape[0]
kcols = kernel.shape[1]
# Figure out relative kernel offsets:
mid_row = krows / 2
mid_col = kcols / 2
offsets = []
for row in range(krows):
for col in range(kcols):
offsets.append((row - mid_row, col - mid_col))
# Build the operator matrix:
K = []
for row in range(nrows):
for col in range(ncols):
K.append([0] * (nrows * ncols))
for offset in offsets:
image_row = row + offset[0]
image_col = col + offset[1]
if (0 <= image_row < nrows) and (0 <= image_col < ncols):
index = image_row * ncols + image_col
kernel_coords = (offset[0] + mid_row, offset[1] + mid_col)
K[-1][index] = kernel[kernel_coords]
print 'Kernel matrix:'
print K
  */
  return operator_matrix;
}

cv::Mat DegradationOperator::GetOperatorMatrix(
    const cv::Size& image_size, const int index) const {

  const int num_pixels = image_size.width * image_size.height;
  return cv::Mat::eye(num_pixels, num_pixels, util::kOpenCvMatrixType);
}

}  // namespace super_resolution
