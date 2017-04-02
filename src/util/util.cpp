#include "util/util.h"

#include <dirent.h>

#include <iostream>
#include <string>
#include <vector>

#include "image/image_data.h"

#include "opencv2/core/core.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

namespace super_resolution {
namespace util {
namespace {

// If true, gflags::ParseCommandLineFlags will remove all flags that it
// processed from the argv list. Any other misc input parameters will remain.
constexpr bool kRemoveFlagsAfterParsing = true;

}  // namespace

void InitApp(int argc, char** argv, const std::string& usage_message) {
  google::SetUsageMessage(usage_message);
  google::SetVersionString(kCodeVersion);
  gflags::ParseCommandLineFlags(&argc, &argv, kRemoveFlagsAfterParsing);
  google::InitGoogleLogging(argv[0]);

  // TODO: remove this or put it under a debug guard.
  FLAGS_logtostderr = true;

  LOG(INFO) << "Running with OpenCV version " << CV_VERSION << ".";
}

std::string GetRootCodeDirectory() {
#ifdef ROOT_CODE_DIRECTORY
  return std::string(ROOT_CODE_DIRECTORY);
#else
  LOG(WARNING) << "ROOT_CODE_DIRECTORY is not defined. "
               << "Returning local Unix directory ('.')";
  return ".";
#endif
}

std::string GetAbsoluteCodePath(const std::string& relative_path) {
  return GetRootCodeDirectory() + "/" + relative_path;
}

std::vector<std::string> ListFilesInDirectory(const std::string& directory) {
  std::vector<std::string> list_of_files;

  // Try to open the directory. If this fails, return an empty list of files.
  DIR* dir_ptr = opendir(directory.c_str());
  if (dir_ptr == NULL) {
    LOG(ERROR) << "Could not open directory: " << directory;
    return list_of_files;
  }

  // Loop through the contents and only keep non-hidden files.
  struct dirent* dirp;
  while ((dirp = readdir(dir_ptr)) != NULL) {
    // Skip directories.
    if (dirp->d_type != DT_REG) {
      continue;
    }
    const std::string item = std::string(dirp->d_name);
    // Skip hidden files.
    if (item.compare(0, 1, ".") == 0) {
      continue;
    }
    list_of_files.push_back(std::string(dirp->d_name));
  }
  closedir(dir_ptr);

  return list_of_files;
}

int GetPixelIndex(
    const cv::Size& image_size,
    const int channel,
    const int row,
    const int col) {

  const int channel_index = channel * (image_size.width * image_size.height);
  return channel_index + (row * image_size.width + col);
}

}  // namespace util
}  // namespace super_resolution
