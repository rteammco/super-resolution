# This script converts a 128 x 128 image into FT-IR data. The image should be
# binary in nature, meaning every pixel is either white (0) or black (1). All
# non-zero values in the image will be converted to 1 before conversion into
# FT-IR test format.

import argparse
import random

from PIL import Image
from PIL import ImageOps

IMAGE_SIZE = 128
NOISE_INTENSITY = 0.2

def convert_image(src_image_path, dest_txt_path):
  """
  """
  # Open the image, invert it, and turn it into binary.
  image = Image.open(src_image_path)
  image = image.convert('L')
  image = ImageOps.invert(image)
  image = image.convert('1')
  # Make 5 separate images, each with different pixel values.
  channels = []
  for i in range(5):
    channels.append(image.copy())
  data = []
  # row => all bands in col 0, all bands in col 1, ....
  for row in range(IMAGE_SIZE):
    data.append([])
    for i in range(len(channels)):
      offset = i * 0.1
      for col in range(IMAGE_SIZE):
        val = channels[i].getpixel((col, row))
        noise = random.uniform(0, NOISE_INTENSITY)
        if val > 0:
          val = 1 - offset
          val = val - noise
        else:
          val = val + noise
        data[row].append(val)
  # Write out the text data file.
  outfile = open(dest_txt_path, 'w')
  for row_data in data:
    row_out = ','.join(map(str, row_data))
    outfile.write(row_out + '\n')
  outfile.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Converts a regular image into a noisy FT-IR test sample.')
  parser.add_argument('--image', required=True, help='The input image file.')
  parser.add_argument('--output', required=True, help='The output text file.')
  args = parser.parse_args()
  convert_image(args.image, args.output)
