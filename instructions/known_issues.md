Known Issues
================

This is a list some common issues (and their solutions) that might cause installation or runtime errors.


OpenCV Error
--------------------

The following OpenCV error might be displayed when trying to run tests or code that displays images:
```
OpenCV Error: Unsupported format or combination of formats () in threshold, file /tmp/opencv3-20160626-61731-v18vfg/opencv-3.1.0/modules/imgproc/src/thresh.cpp, line 1273
```
This happens due to a bug in an older version of OpenCV 3 where the `cv::threshold` function does not support 64-bit floats (doubles) in images.

The fix is to update your OpenCV installation.

#### macOS

```
brew update
brew upgrade opencv3
```

#### Linux

Probably need to download and install the latest version again. Sorry :(
