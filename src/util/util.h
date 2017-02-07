// Contains general utilities for the codebase.

#ifndef SRC_UTIL_UTIL_H_
#define SRC_UTIL_UTIL_H_

#include <string>
#include <vector>

#include "image/image_data.h"

namespace super_resolution {
namespace util {

const char kCodeVersion[] = "0.1";

// Initializes the app. Processes all of the command line arguments with gflags
// and initializes logging with glog. Sets the usage message and app version.
void InitApp(int argc, char** argv, const std::string& usage_message = "");

// Returns a list of all files in the given directory. If no files are present,
// returns an empty list. Subdirectories and hidden files are not included in
// the listing.
std::vector<std::string> ListFilesInDirectory(const std::string& directory);

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

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_UTIL_H_
