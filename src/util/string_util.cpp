#include "util/string_util.h"

#include <functional>
#include <string>
#include <vector>

namespace super_resolution {
namespace util {

std::vector<std::string> SplitString(
    const std::string& whole_string,
    const char delimiter,
    const bool ignore_empty_pieces,
    const int max_num_pieces) {

  // For a single piece, just return a list containing the original string.
  if (max_num_pieces == 1) {
    return {whole_string};
  }

  std::vector<std::string> parts;
  std::string remaining = whole_string;
  int split_position = remaining.find(delimiter);
  int num_parts_added = 0;
  while (split_position >= 0) {
    const std::string part = remaining.substr(0, split_position);
    if (!ignore_empty_pieces || !part.empty()) {
      parts.push_back(part);
      num_parts_added += 1;
    }
    remaining = remaining.substr(split_position + 1);
    split_position = remaining.find(delimiter);
    if (max_num_pieces > 1 && (parts.size() + 1) >= max_num_pieces) {
      break;
    }
  }
  if (!ignore_empty_pieces || !remaining.empty()) {
    parts.push_back(remaining);
  }

  return parts;
}

// Inspired from:
// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
std::string TrimString(const std::string& untrimmed_string) {
  std::string trimmed_string = untrimmed_string;
  // Trim left.
  trimmed_string.erase(trimmed_string.begin(), std::find_if(
      trimmed_string.begin(),
      trimmed_string.end(),
      std::not1(std::ptr_fun<int, int>(std::isspace))));
  // Trim right.
  trimmed_string.erase(
      std::find_if(
          trimmed_string.rbegin(),
          trimmed_string.rend(),
          std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
      trimmed_string.end());
  return trimmed_string;
}

std::string GetFileExtension(const std::string& file_path) {
  std::string::size_type pos = file_path.rfind(".");
  if (pos == std::string::npos) {
    return "";
  }
  return file_path.substr(pos + 1);
}

}  // namespace util
}  // namespace super_resolution
