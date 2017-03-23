// General helper functions related to string parsing and processing.

#ifndef SRC_UTIL_STRING_UTIL_H_
#define SRC_UTIL_STRING_UTIL_H_

#include <string>
#include <vector>

namespace super_resolution {
namespace util {

// Splits a string around the given delimiter into one or more pieces. If the
// given string contains a continuous sequence of two or more delimiters, this
// will result in empty strings being returned in the split unless
// ignore_empty_pieces is set to true.
//
// If max_num_pieces is positive, the string will be split around the delimiter
// left-to-right for a maximum of max_num_pieces times.
//
// Examples:
//   SplitString("true = false", '=') => {"true ", " false"}
//   SplitString(" x y z", ' ') => {"", "x", "y", "z"}
//   SplitString(" x y z", ' ', true) => {"x", "y", "z"}
//   SplitString(" x y z", ' ', true, 2) => {"x", "y z"}
//   SplitString(" x y z", ' ', false, 3) => {"", "x", "y z"}
std::vector<std::string> SplitString(
    const std::string& whole_string,
    const char delimiter = ' ',
    const bool ignore_empty_pieces = false,
    const int max_num_pieces = 0);

// Returns a trimmed version of the given untrimmed string, where all white
// space (including newlines) will be removed from the left and right edges.
//
// Examples:
//   TrimString("   hello\n") => "hello"
//   TrimString("lol") => "lol"
std::string TrimString(const std::string& untrimmed_string);

}  // namespace util
}  // namespace super_resolution

#endif  // SRC_UTIL_STRING_UTIL_H_
