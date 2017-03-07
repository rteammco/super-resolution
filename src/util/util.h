// Contains general utilities for the codebase.

#ifndef SRC_UTIL_UTIL_H_
#define SRC_UTIL_UTIL_H_

#include <string>
#include <vector>

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

namespace super_resolution {
namespace util {

const char kCodeVersion[] = "0.1";

// Initializes the app. Processes all of the command line arguments with gflags
// and initializes logging with glog. Sets the usage message and app version.
void InitApp(int argc, char** argv, const std::string& usage_message = "");

// Returns the root directory where this project was compiled. This uses the
// root path preprocessor macro specified by CMake. If for some reason this
// flag isn't defined, a warning will be logged and the local directory (".")
// will be returned instead.
std::string GetRootCodeDirectory();

// Returns the absolute path on the computer this code was compiled on of given
// relative path within the root code directory. For example,
//   GetAbsoluteCodePath("src/super_resolution.cpp")
// would return, for example,
//   "/Users/richard/Code/SuperResolution/src/super_resolution.cpp".
// Requires compilation using the provided CMake file.
std::string GetAbsoluteCodePath(const std::string& relative_path);

// Returns a list of all files in the given directory. If no files are present,
// returns an empty list. Subdirectories and hidden files are not included in
// the listing.
std::vector<std::string> ListFilesInDirectory(const std::string& directory);

// Splits a string around the given delimiter into one or more pieces. If the
// given string contains a continuous sequence of two or more delimiters, this
// will result in empty strings being returned in the split unless
// ignore_empty_pieces is set to true.
//
// Examples:
//   SplitString("true = false", '=') => {"true ", " false"}
//   SplitString(" x y z", ' ') => {"", "x", "y", "z"}
//   SplitString(" x y z", ' ', true) => {"x", "y", "z"}
std::vector<std::string> SplitString(
    const std::string& whole_string,
    const char delimiter = ' ',
    const bool ignore_empty_pieces = false);

// Returns a trimmed version of the given untrimmed string, where all white
// space (including newlines) will be removed from the left and right edges.
//
// Examples:
//   TrimString("   hello\n") => "hello"
//   TrimString("lol") => "lol"
std::string TrimString(const std::string& untrimmed_string);

// Displays a given image until the user presses any key to close the window.
// If rescale is set to true, the image will be resized (without interpolation)
// if it is smaller than the predefined minimum size. Larger images will always
// automatically be scaled to fit the screen.
void DisplayImage(
    const ImageData& image,
    const std::string& window_name = "Image",
    const bool rescale = true);

// Displays multiple images side-by-side in the same way as DisplayImage.
void DisplayImagesSideBySide(
    const std::vector<ImageData>& images,
    const std::string& window_name = "Images",
    const bool rescale = true);

// Returns the index into a pixel array given its channel (band), row, and
// column coordinates. This assumes the standard channel-row-col ordering on an
// array containing image data.
int GetPixelIndex(
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col);

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_UTIL_H_
