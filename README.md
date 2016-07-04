Install
===

OS X
---

First install Homebrew.

Install required libraries:
  $ brew install gflags
  $ brew install glog
  $ brew install opencv3
#  $ brew install ffmpeg ??
#  $ brew install ceres-solver
#  $ brew install vtk
  glog: follow instructions here: https://github.com/google/glog

Make sure "/usr/local/lib" is in your library path:
  export LIBRARY\_PATH=/usr/local/lib

To cmake with OpenCV, need to run export OpenCV\_DIR=/usr/local/opt/opencv3 before running cmake.
