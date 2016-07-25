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

  // TODO(richard): remove this or put it under a debug guard.
  FLAGS_logtostderr = true;
}

}  // namespace util
}  // namespace super_resolution
