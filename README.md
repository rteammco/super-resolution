Super Resolution
================

Install Instructions
--------------------

#### OS X

First install Homebrew: http://brew.sh/.
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Install required libraries:
```
brew install gflags
brew install glog
brew install opencv3
brew install webp
```

Make sure `/usr/local/lib` is in your library path:
```
export LIBRARY_PATH=/usr/local/lib
```
or, if you have Xcode, installing the command line tools will work:
```
xcode-select install
```

To cmake with OpenCV3, need to run
```
export OpenCV_DIR=/usr/local/opt/opencv3
```
before running cmake.

Then the standard CMake process:
```
mkdir build && cd build
cmake ..
make
```

Directory Structure
--------------------
Add source files are in `./src`. Most files (classes and utilities) are organized into subdirectories. All files that are compiled into binaries (i.e. "main" files) are in the top level of `./src`.

Tests are included in `./test` and follow a similar directory structure.
