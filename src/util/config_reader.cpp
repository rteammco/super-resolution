#include "util/config_reader.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/string_util.h"
#include "util/util.h"

#include "glog/logging.h"

namespace super_resolution {
namespace util {

void ConfigurationFileReader::ReadFromFile(const std::string& file_path) {
  std::ifstream fin(file_path);
  CHECK(fin.is_open())
      << "Could not open file '" << file_path << "' for reading.";

  std::string line;
  while (std::getline(fin, line)) {
    if (line.find("#") == 0) {  // If string starts with a "#" it's a comment.
      continue;
    }
    const std::vector<std::string> parts =
        SplitString(line, key_value_delimiter_, true, 2);
    if (parts.size() != 2) {
      continue;
    }
    const std::string key = TrimString(parts[0]);
    const std::string value = TrimString(parts[1]);
    config_map_[key] = value;
  }
  fin.close();
}

bool ConfigurationFileReader::HasValue(const std::string& key) const {
  return config_map_.find(key) != config_map_.end();
}

std::string ConfigurationFileReader::GetValue(const std::string& key) const {
  const auto iterator = config_map_.find(key);
  if (iterator != config_map_.end()) {
    return iterator->second;
  }
  return "";
}

int ConfigurationFileReader::GetValueAsInt(const std::string& key) const {
  if (!HasValue(key)) {
    LOG(WARNING)
        << "Value for key '" << key << "' does not exist. Returning 0.";
    return 0;
  }
  const std::string value = GetValueOrDie(key);
  return std::atoi(value.c_str());  // atoi returns 0 if invalid.
}

std::string ConfigurationFileReader::GetValueOrDie(
    const std::string& key) const {

  CHECK(HasValue(key))
      << "The map does not have a value for key '" << key << "'.";
  return GetValue(key);
}

}  // namespace util
}  // namespace super_resolution
