#!/usr/bin/env python

import argparse
import subprocess

# TODO: this is temporary and should be filled in using a GUI or config file.
configuration = {
  'scale': 2,
  'blur_radius': 5,
  'blur_sigma': 0.7,
  'noise_sigma': 0.0,
  'number_of_frames': 4,
  'hr_image_path': '../test_data/fb.png',
  'lr_image_dir': '../test_data/OUT',
  'motion_sequence_path': '../test_data/test_motion_sequence_4.txt',
  'display_mode': 'compare'
}

def run_generate_data(binary_path, config):
  """ Runs the GenerateData binary.
  """
  command = binary_path + '/GenerateData'
  command += ' --input_image={}'.format(config['hr_image_path'])
  command += ' --output_image_dir={}'.format(config['lr_image_dir'])
  command += ' --motion_sequence_path={}'.format(config['motion_sequence_path'])
  command += ' --blur_radius={}'.format(config['blur_radius'])
  command += ' --blur_sigma={}'.format(config['blur_sigma'])
  command += ' --noise_sigma={}'.format(config['noise_sigma'])
  command += ' --downsampling_scale={}'.format(config['scale'])
  command += ' --number_of_frames={}'.format(config['number_of_frames'])
  print 'Running GenerateData command:'
  print command
  subprocess.call(command.split(' '))

def run_super_resolution(binary_path, config):
  """ Runs the SuperResolution binary.
  """
  command = binary_path + '/SuperResolution'
  command += ' --data_path={}'.format(config['lr_image_dir'])
  command += ' --upsampling_scale={}'.format(config['scale'])
  command += ' --blur_radius={}'.format(config['blur_radius'])
  command += ' --blur_sigma={}'.format(config['blur_sigma'])
  command += ' --motion_sequence_path={}'.format(config['motion_sequence_path'])
  command += ' --display_mode={}'.format(config['display_mode'])
  print 'Running SuperResolution command:'
  print command
  subprocess.call(command.split(' '))

if __name__ == '__main__':
  # Run the appropriate binary based on the --binary argument.
  parser = argparse.ArgumentParser(
      description='Run script for super-resolution binaries.')
  parser.add_argument('--binary', required=True,
      help='The binary to run (SuperResolution or GenerateData).')
  args = parser.parse_args()
  binary_path = 'bin'
  if args.binary == 'SuperResolution':
    run_super_resolution(binary_path, configuration)
  elif args.bianry == 'GenerateData':
    run_generate_data(binary_path, configuration)
