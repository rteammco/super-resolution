#include "util/config_reader.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/util.h"

#include "glog/logging.h"

namespace super_resolution {
namespace util {

std::unordered_map<std::string, std::string> ReadConfigurationFile(
    const std::string& config_file_path, const char key_value_delimiter) {

  std::ifstream fin(config_file_path);
  CHECK(fin.is_open()) << "Could not open file '" << config_file_path << "'.";

  std::unordered_map<std::string, std::string> config_map;
  std::string line;
  while (std::getline(fin, line)) {
    if (line.find("#") == 0) {  // If string starts with a "#" it's a comment.
      continue;
    }
    const std::vector<std::string> parts =
        SplitString(line, key_value_delimiter, true);
    if (parts.size() != 2) {
      continue;
    }
    const std::string key = TrimString(parts[0]);
    const std::string value = TrimString(parts[1]);
    config_map[key] = value;
  }
  fin.close();

  return config_map;
}

}  // namespace util
}  // namespace super_resolution
