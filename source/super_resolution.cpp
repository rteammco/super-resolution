#include <iostream>

#include "gflags/gflags.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "Hello, world" << std::endl;

  return EXIT_SUCCESS;
}
