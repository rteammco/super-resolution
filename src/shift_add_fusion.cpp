// Contains code for the shift-add fusion algorithm as explained in "An
// Introduction to Super-Resolution Imaging (2012)".

#include <string>

#include "util/macros.h"
#include "util/util.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv,
      "A trivial implementation of shift-add fusion.");

  std::string FLAGS_my_str = "";

  REQUIRE_ARG(FLAGS_my_str);

  return EXIT_SUCCESS;
}
