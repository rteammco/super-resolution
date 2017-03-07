// Provides a standard way of reading configuration files which define
// key/value pairs for parameters.

#ifndef SRC_UTIL_CONFIG_READER_H_
#define SRC_UTIL_CONFIG_READER_H_

#include <string>
#include <unordered_map>

namespace super_resolution {
namespace util {

// Returns an unordered_map that maps configuration keys to their assigned
// values as strings. This function assumes the given config file contains
// key/value pairs, one per line, delimited by the given delimiter. For
// example, to specify a file path in the configuration file, add a line as
// follows:
//   file  /usr/bin/secret.txt
// or (for delimiter '='):
//   item_count = 10
// Empty lines and lines which start with a "#" will be ignored.
std::unordered_map<std::string, std::string> ReadConfigurationFile(
    const std::string& config_file_path, const char key_value_delimiter = ' ');

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_CONFIG_READER_H_
