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

if __name__ == '__main__':
  I  = np.matrix(
      [[1, 2, 3, 4, 5, 6],
       [7, 8, 9, 0, 1, 2],
       [9, 7, 5, 4, 2, 1],
       [2, 4, 6, 8, 0, 1]])
  print 'Test image:'
  print I
  downsample(I, 2)
