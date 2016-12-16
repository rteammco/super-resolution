# Matrix tests.

import numpy as np

def downsample(image, scale):
  """Downsamples the image given the scale."""
  nrows = image.shape[0]
  ncols = image.shape[1]
  Iv = image.reshape(nrows * ncols, 1)  # vectorize the image
  # Build the downsampling matrix:
  D = []
  for row in range(nrows):
    if row % scale != 0:
      continue
    for col in range(ncols):
      if col % scale == 0:
        D.append([0] * (nrows * ncols))
        index = row * ncols + col
        D[-1][index] = 1
  D = np.matrix(D)
  print 'Downsampling matrix:'
  print D
  print D.shape
  # Apply the downsampling and print the results:
  DIv = D * Iv  # downsampled, vectorized
  print 'Final vectorized output:'
  print DIv
  DI = DIv.reshape(nrows / scale, ncols / scale)
  print 'Final downsampled image:'
  print DI
  return DI

def conv_kernel(image, kernel):
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
  print 'Convolved image:'
  Iv = image.reshape(nrows * ncols, 1)  # vectorize the image
  KIv = K * Iv
  KI = KIv.reshape(nrows, ncols)  # un-vectorize the image back
  print KI
  print 'Tranposed kernel matrix:'
  KT = np.transpose(K)
  print KT
  print 'Tranpose convolved image:'
  KTIv = KT * Iv
  KTI = KTIv.reshape(nrows, ncols)
  print KTI

if __name__ == '__main__':
  I = np.matrix(
      [[1, 2, 3, 4, 5, 6],
       [7, 8, 9, 0, 1, 2],
       [9, 7, 5, 4, 2, 1],
       [2, 4, 6, 8, 0, 1]])
  print 'Test image:'
  print I
  I = downsample(I, 2)
  # Gradient operator kernel:
  #kernel = np.matrix(
  #    [[-1, 0, 1],
  #     [-2, 0, 2],
  #     [-1, 0, 1]])
  # Gaussian blur kernel:
  #kernel = np.matrix(
  #    [[0.0625, 0.1250, 0.0625],
  #     [0.1250, 0.2500, 0.1250],
  #     [0.0625, 0.1250, 0.0625]])
  # Test tranpose of two kernels:
  kernel = np.matrix(
      [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
  conv_kernel(I, kernel)
  print '---------------'
  kernel2 = np.matrix(
      [[9, 8, 7],
       [6, 5, 4],
       [3, 2, 1]])
  conv_kernel(I, kernel2)
