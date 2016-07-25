// Contains general utilities for the codebase.

#ifndef SRC_UTIL_UTIL_H_
#define SRC_UTIL_UTIL_H_

#include <string>

namespace super_resolution {
namespace util {

const char kCodeVersion[] = "0.1";

// Initializes the app. Processes all of the command line arguments with gflags
// and initializes logging with glog. Sets the usage message and app version.
void InitApp(int argc, char** argv, const std::string& usage_message = "");

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_UTIL_H_
