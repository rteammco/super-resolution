#include <string>
#include <unordered_map>
#include <vector>

#include "util/config_reader.h"
#include "util/util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::util::GetAbsoluteCodePath;
using testing::ElementsAre;
using testing::UnorderedElementsAreArray;
using testing::Pair;

// Path to a test config file (to test ReadConfigurationFile).
static const std::string kTestConfigFilePath =
    GetAbsoluteCodePath("test_data/test_hs_config.txt");

TEST(Util, SplitString) {
  EXPECT_THAT(
      super_resolution::util::SplitString("a b c"),
      ElementsAre("a", "b", "c"));
  EXPECT_THAT(
      super_resolution::util::SplitString("true = false", '='),
      ElementsAre("true ", " false"));
  EXPECT_THAT(
      super_resolution::util::SplitString("  hi how are  u? ", ' '),
      ElementsAre("", "", "hi", "how", "are", "", "u?", ""));
  EXPECT_THAT(
      super_resolution::util::SplitString("  hi how are  u? ", ' ', true),
      ElementsAre("hi", "how", "are", "u?"));
  EXPECT_THAT(
      super_resolution::util::SplitString(""),
      ElementsAre(""));
}

TEST(Util, TrimString) {
  EXPECT_EQ(super_resolution::util::TrimString(
      "  one two three  "),
      "one two three");
  EXPECT_EQ(super_resolution::util::TrimString(
      "\nomg \tthis is the best\n"),
      "omg \tthis is the best");
  EXPECT_EQ(super_resolution::util::TrimString(
      " \tSPAAAAAAAAAAAACE      SPAAAACE\n"),
      "SPAAAAAAAAAAAACE      SPAAAACE");
  EXPECT_EQ(super_resolution::util::TrimString(
      "asdf"),
      "asdf");
  EXPECT_EQ(super_resolution::util::TrimString(
      "\n"),
      "");
  EXPECT_EQ(super_resolution::util::TrimString(
      "  \n  "),
      "");
}

TEST(Util, ReadConfigurationFile) {
  const std::unordered_map<std::string, std::string> config_map =
      super_resolution::util::ReadConfigurationFile(kTestConfigFilePath);
  EXPECT_THAT(config_map, UnorderedElementsAreArray({
      Pair("file", "../test_data/example_envi_data"),
      Pair("interleave", "bsq"),
      Pair("data_type", "float"),
      Pair("big_endian", "false"),
      Pair("header_offset", "0"),
      Pair("num_data_rows", "9"),
      Pair("num_data_cols", "5"),
      Pair("num_data_bands", "10"),
      Pair("start_row", "2"),
      Pair("end_row", "8"),
      Pair("start_col", "0"),
      Pair("end_col", "3"),
      Pair("start_band", "5"),
      Pair("end_band", "10")}));
}
