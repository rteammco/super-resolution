Super Resolution
================

Install Instructions
--------------------

#### OS X

First install Homebrew.

Install required libraries:
```
brew install gflags
brew install glog
brew install opencv3
```

Make sure `/usr/local/lib` is in your library path:
```
export LIBRARY_PATH=/usr/local/lib
```

To cmake with OpenCV, need to run
```
export OpenCV_DIR=/usr/local/opt/opencv3
```
before running cmake.
