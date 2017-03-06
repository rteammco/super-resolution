#include <string>
#include <vector>

#include "util/util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

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
