#include <string>
#include <unordered_map>
#include <vector>

#include "util/config_reader.h"
#include "util/string_util.h"
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
      super_resolution::util::SplitString("  hi how are  u? ", ' ', true, 3),
      ElementsAre("hi", "how", "are  u? "));
  EXPECT_THAT(
      super_resolution::util::SplitString("  hi how are  u? ", ' ', false, 4),
      ElementsAre("", "", "hi", "how are  u? "));
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
  super_resolution::util::ConfigurationFileReader config_reader;
  config_reader.ReadFromFile(kTestConfigFilePath);
  EXPECT_EQ(config_reader.GetValue("file"), "../test_data/example_envi_data");
  EXPECT_EQ(config_reader.GetValue("interleave"), "bsq");
  EXPECT_EQ(config_reader.GetValue("data_type"), "float");
  EXPECT_EQ(config_reader.GetValue("big_endian"), "false");
  EXPECT_EQ(config_reader.GetValue("header_offset"), "0");
  EXPECT_EQ(config_reader.GetValue("num_data_rows"), "9");
  EXPECT_EQ(config_reader.GetValueAsInt("num_data_cols"), 5);
  EXPECT_EQ(config_reader.GetValue("num_data_bands"), "10");
  EXPECT_EQ(config_reader.GetValueAsInt("start_row"), 2);
  EXPECT_EQ(config_reader.GetValueAsInt("end_row"), 8);
  EXPECT_EQ(config_reader.GetValue("start_col"), "0");
  EXPECT_EQ(config_reader.GetValue("end_col"), "3");
  EXPECT_EQ(config_reader.GetValue("start_band"), "5");
  EXPECT_EQ(config_reader.GetValue("end_band"), "10");
}
