// Contains specific utility macros for the codebase.

#ifndef SRC_UTIL_MACROS_H_
#define SRC_UTIL_MACROS_H_

#include <string>

#include "glog/logging.h"

// Call on an argument flag from gflags to set it as required. If the argument
// is missing, the program will die with an error. The variable name must be
// as defined by gflags; i.e. of the form "FLAGS_x".
//
// Usage example:
//   REQUIRE_ARG(FLAGS_file_path);
//
#define REQUIRE_ARG(arg_string) \
    LOG_IF(FATAL, GOOGLE_PREDICT_BRANCH_NOT_TAKEN(arg_string.empty())) \
        << "Missing required argument: " \
        << std::string(#arg_string).substr(6);

#endif  // SRC_UTIL_MACROS_H_
