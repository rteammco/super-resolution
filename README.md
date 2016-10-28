Super Resolution
================

Install Instructions
--------------------

#### OS X

First install Homebrew: http://brew.sh/.
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
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
brew install webp
brew install ceres-solver
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

Make sure `/usr/local/lib` is in your library path. If you have Xcode installed, then run
```
xcode-select install
```

or otherwise, add this to your .bash_profile:
```
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

Directory Structure
--------------------
Add source files are in `./src`. Most files (classes and utilities) are organized into subdirectories. All files that are compiled into binaries (i.e. "main" files) are in the top level of `./src`.

Tests are included in `./test` and follow a similar directory structure.
