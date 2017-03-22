// Provides a standard way of reading configuration files which define
// key/value pairs for parameters.

#ifndef SRC_UTIL_CONFIG_READER_H_
#define SRC_UTIL_CONFIG_READER_H_

#include <string>
#include <unordered_map>

namespace super_resolution {
namespace util {

class ConfigurationFileIO {
 public:
  // Reads the configuration data from the given file. An error will occur if
  // the given file does not exist or cannot be read.
  void ReadFromFile(const std::string& file_path);

  // Writes the configuration data to the given file. An error will occur if
  // the given file cannot be written to.
  void WriteToFile(const std::string& file_path) const;

  // Set the delimiter for file reading or writing. This delimiter will
  // determine how key-value pairs are separated on each line of the data. If
  // the file is being written, this is the delimiter that will be used to
  // separate the values.
  void SetDelimiter(const char delimiter) {
    key_value_delimiter_ = delimiter;
  }

  // Sets the value for the given key. If the value did not previously exist,
  // it will be added. Otherwise the existing value will be modified.
  void SetValue(const std::string& key, const std::string& value) {
    config_map_[key] = value;
  }

  // Returns true if a value exists in the configuration for the given key.
  bool HasValue(const std::string& key) const;

  // Returns the value for the given key. If the given key does not map to any
  // value, an empty string will be returned and a warning will be logged.
  std::string GetValue(const std::string& key) const;

  // Returns the given value, interpreted as an int. If the value does not
  // exist or cannot be interpreted as an int, a default value of 0 will be
  // returned.
  int GetValueAsInt(const std::string& key) const;

  // Returns the value for the given key. If the given key does not map to any
  // value, this will result in a fatal error, and the program will terminate.
  std::string GetValueOrDie(const std::string& key) const;

 private:
  // The delimiter used to separate keys and values in the file. Set this for
  // reading or writing the configuration files.
  char key_value_delimiter_ = ' ';

  // The map containing key-value pairs. These will either be loaded from a
  // file, set by the user, and/or written out to a file.
  std::unordered_map<std::string, std::string> config_map_;
};

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_CONFIG_READER_H_
