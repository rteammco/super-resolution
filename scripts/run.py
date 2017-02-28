#!/usr/bin/env python

import argparse
import subprocess

# TODO: this is temporary and should be filled in using a GUI or config file.
configuration = {
  'scale': 2,
  'blur_radius': 3,
  'blur_sigma': 0.5,
  'noise_sigma': 5.0,
#  'hr_image_path': '../test_data/fb.png',
  'hr_image_path': '../test_data/dallas_qtr.jpg',
  'lr_image_dir': '../test_data/OUT',
  'number_of_frames': 9,
  'motion_sequence_path': '../test_data/test_motion_sequence_9.txt',
  # Solver-only options:
  'interpolate_color': True,
  'solve_in_wavelet_domain': False,
  'regularizer': 'tv',
  'regularization_parameter': 0.001,
  'solver': 'lbfgs',
  'solver_iterations': 50,  # = 0 infinite
  'use_numerical_differentiation': False,
  'display_mode': 'compare',
  'generate_lr_images': True,  # for SR binary testing
  'verbose_solver': True,  # for SR binary testing
  'evaluator': 'psnr'
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
  command += ' --upsampling_scale={}'.format(config['scale'])
  command += ' --blur_radius={}'.format(config['blur_radius'])
  command += ' --blur_sigma={}'.format(config['blur_sigma'])
  command += ' --motion_sequence_path={}'.format(config['motion_sequence_path'])
  command += ' --regularizer={}'.format(config['regularizer'])
  command += ' --regularization_parameter={}'.format(
      config['regularization_parameter'])
  command += ' --display_mode={}'.format(config['display_mode'])
  command += ' --solver={}'.format(config['solver'])
  command += ' --solver_iterations={}'.format(config['solver_iterations'])
  if config['generate_lr_images']:
    command += ' --generate_lr_images'
    command += ' --number_of_frames={}'.format(config['number_of_frames'])
    command += ' --noise_sigma={}'.format(config['noise_sigma'])
    command += ' --data_path={}'.format(config['hr_image_path'])
    command += ' --evaluator={}'.format(config['evaluator'])
  else:
    command += ' --data_path={}'.format(config['lr_image_dir'])
  if config['interpolate_color']:
    command += ' --interpolate_color'
  if config['solve_in_wavelet_domain']:
    command += ' --solve_in_wavelet_domain'
  if config['verbose_solver']:
     command += ' --verbose'
  if config['use_numerical_differentiation']:
    command += ' --use_numerical_differentiation'
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
  elif args.binary == 'GenerateData':
    run_generate_data(binary_path, configuration)
