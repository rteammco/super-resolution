#include "util/util.h"

#include <dirent.h>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

namespace super_resolution {
namespace util {

void InitApp(int argc, char** argv, const std::string& usage_message) {
  google::SetUsageMessage(usage_message);
  google::SetVersionString(kCodeVersion);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // TODO: remove this or put it under a debug guard.
  FLAGS_logtostderr = true;
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

}  // namespace util
}  // namespace super_resolution
