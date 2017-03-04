#!/usr/bin/env python

import argparse
import subprocess
import Tkinter as tk

# The default configuration values.
# TODO: This should be loaded from a config file.
configuration = {
  'scale': 2,
  'blur_radius': 3,
  'blur_sigma': 0.5,
  'noise_sigma': 5.0,
#  'hr_image_path': '../test_data/fb.png',
  'hr_image_path': '../test_data/dallas_half.jpg',
  'lr_image_dir': '../test_data/OUT',
  'number_of_frames': 9,
  'motion_sequence_path': '../test_data/test_motion_sequence_9.txt',
  # Solver-only options:
  'interpolate_color': True,
  'solve_in_wavelet_domain': False,
  'regularizer': 'btv',
  'regularization_parameter': 0.01,
  'btv_scale_range': 3,
  'btv_spatial_decay': 0.5,
  'solver': 'cg',
  'solver_iterations': 50,  # = 0 infinite
  'use_numerical_differentiation': False,
  'display_mode': 'compare',
  'generate_lr_images': True,  # for SR binary testing
  'verbose_solver': True,  # for SR binary testing
  'evaluator': 'psnr'
}

def run_generate_data(binary_path, config):
  """ Runs the GenerateData binary.

  Args:
    binary_path: The path to the GenerateData executable binary.
    config: The configuration dictionary (see default configuration above).
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

  Args:
    binary_path: The path to the SuperResolution executable binary.
    config: The configuration dictionary (see default configuration above).
  """
  command = binary_path + '/SuperResolution'
  command += ' --upsampling_scale={}'.format(config['scale'])
  command += ' --blur_radius={}'.format(config['blur_radius'])
  command += ' --blur_sigma={}'.format(config['blur_sigma'])
  command += ' --motion_sequence_path={}'.format(config['motion_sequence_path'])
  command += ' --regularizer={}'.format(config['regularizer'])
  if config['regularizer'] == 'btv':
    command += ' --btv_scale_range={}'.format(config['btv_scale_range'])
    command += ' --btv_spatial_decay={}'.format(config['btv_spatial_decay'])
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

def make_gui_window(binary_path):
  """ Creates a Tkinter window so the user can set input values.

  Args:
    binary_path: The path to the executable binaries to be run.
  """
  gui_window = tk.Tk()
  gui_window.title('Run Super-Resolution')
  # Make the input fields:
  next_row = 0
  for parameter in configuration:
    tk.Label(gui_window, text = parameter).grid(row = next_row)
    def text_change_callback(key, sv):
      """ When the StringVar gets changed, update the configuration. """
      val = sv.get()
      if val.lower() == 'true':
        val = True
      elif val.lower() == 'false':
        val = False
      configuration[key] = val
    sv = tk.StringVar()
    sv.trace('w', lambda name, index, mode, key = parameter, sv = sv:
        text_change_callback(key, sv))
    sv.set(str(configuration[parameter]))
    entry = tk.Entry(gui_window, textvariable = sv)
    entry.grid(row = next_row, column = 1)
    next_row += 1
  # Make the run buttons:
  tk.Button(
      gui_window,
      text='SuperResolution',
      command = lambda: run_super_resolution(binary_path, configuration)).grid(
          row = next_row, column = 0, sticky = tk.W, pady = 10, padx = 10)
  tk.Button(
      gui_window,
      text='GenerateData',
      command = lambda: run_generate_data(binary_path, configuration)).grid(
          row = next_row, column = 1, sticky = tk.W, pady = 10, padx = 10)
  tk.mainloop()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run script for super-resolution binaries.')
  parser.add_argument('--binary', required=False,
      help='The binary to run (SuperResolution or GenerateData).')
  args = parser.parse_args()
  binary_path = 'bin'
  # Run the appropriate binary based on the --binary argument.
  if args.binary:
    if args.binary == 'SuperResolution':
      run_super_resolution(binary_path, configuration)
    elif args.binary == 'GenerateData':
      run_generate_data(binary_path, configuration)
  # Otherwise open a GUI and let the user select run parameters.
  else:
    make_gui_window(binary_path)
