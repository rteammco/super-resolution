Super Resolution
================

<i><b>NOTE:</b> This project is work-in-progress. It is not currently in a stable release state.</i>

Super-resolution is the process of increasing the resolvability of details in an image.
This work-in-progress code is intended to be a framework for multiframe super-resolution, which combines multiple overlapping low-resolution images into a single higher-resolution output.
The goal is to support both ordinary images (grayscale, RGB) as well as hyperspectral image data which may contain hundreds of channels.

Install Instructions
--------------------

#### macOS

First install Homebrew: http://brew.sh/.
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

When in doubt and just in case,
```
brew update
```

Make sure you have CMake:
```
brew install cmake
```

Install required libraries with Homebrew:
```
brew install gflags
brew install glog
brew install opencv3
```

Install gtest:
```
git clone https://github.com/google/googletest.git
cd googletest/
mkdir build
cd build
cmake ..
make
make install
```

Make sure `/usr/local/lib` is in your library path. If you have Xcode installed, then if you haven't already, run
```
xcode-select --install
```

or otherwise, you may need to add this to your .bash_profile:
```
export CPLUS_INCLUDE_PATH=/usr/local/include
export LIBRARY_PATH=/usr/local/lib
```

To cmake with OpenCV3, you may need to run
```
export OpenCV_DIR=/usr/local/opt/opencv3
```
before running cmake. You only have to do this the first time when running cmake. You can also add this to .bash_profile.

Then the standard CMake process:
```
mkdir build && cd build
cmake ..
make
```

#### Linux (Ubuntu)

To install on Linux without root, the notes in `instructions/install_linux_no_root.txt` may be helpful.

Make sure you have CMake:
```
sudo apt-get install cmake
```

Install required libraries with apt-get:
```
sudo apt-get install libgflags-dev
sudo apt-get install libgoogle-glog-dev
```

Install OpenCV 3 by following the instructions here: http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html.
Also probably install the optional stuff.

Install gtest:
```
git clone https://github.com/google/googletest.git
cd googletest/
mkdir build
cd build
cmake  -DBUILD_SHARED_LIBS=ON ..
make
sudo make install
```

Then the standard CMake process:
```
mkdir build && cd build
cmake ..
make
```

Build and Run
--------------------

<b>NOTE:</b> This is still a work-in-progress project, so the main binaries do not work yet, for the most part.
All code is tested, including cases with real data, through the test framework.

To build the project, make a build directory and run `cmake` followed by `make`:
```
mkdir build && cd build
cmake ..
make
```
It's okay if it doesn't find OpenMP (e.g. as is the case with the default compiler on macOS).

Once everything compiles, make sure it works by running the unit tests. From your `build` directory, or whatever you named it:
```
bin/Test
```

Directory Structure
--------------------
Add source files are in `./src`. Most files (classes and utilities) are organized into subdirectories. All files that are compiled into binaries (i.e. "main" files) are in the top level of `./src`.

Tests are included in `./test` and follow a similar directory structure.

The `./scripts` directory contains simple test or data generation scripts.

The `./test_data` directory contains sample data used by the unit tests and otherwise for testing and experimentation.
