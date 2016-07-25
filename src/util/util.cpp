#include "util/util.h"

#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"

namespace super_resolution {
namespace util {

void InitApp(int argc, char** argv, const std::string& usage_message) {
  google::SetUsageMessage(usage_message);
  google::SetVersionString(kCodeVersion);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
}

}  // namespace util
}  // namespace super_resolution
