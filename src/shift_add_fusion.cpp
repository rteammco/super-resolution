// Contains code for the shift-add fusion algorithm as explained in "An
// Introduction to Super-Resolution Imaging (2012)".

#include <string>

#include "util/util.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
  super_resolution::util::InitApp(argc, argv,
      "A trivial implementation of shift-add fusion.");

  return EXIT_SUCCESS;
}
