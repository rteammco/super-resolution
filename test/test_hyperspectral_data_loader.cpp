#include <string>

#include "hyperspectral/hyperspectral_data_loader.h"
#include "image/image_data.h"
#include "util/test_util.h"
#include "util/util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using super_resolution::test::AreImagesEqual;
using super_resolution::test::AreMatricesEqual;
using super_resolution::util::GetAbsoluteCodePath;

// The numerical error tolerance allowed for the loss of converting floats to
// double precision during the data reading process.
constexpr double kPrecisionErrorTolerance = 1e-6;

// An ENVI header for testing the header reading method.
static const std::string kTestHeaderPath =
    GetAbsoluteCodePath("test_data/example_envi_header.hdr");

// The path the the HS configuration file.
static const std::string kTestConfigFilePath =
    GetAbsoluteCodePath("test_data/test_hs_config.txt");

static const std::string kTestOutputFilePath =
    GetAbsoluteCodePath("test_data/test_tmp_dir/hs_data_loader_envi_out");

// This test verifies that the HSIBinaryDataParameters::ReadHeaderFromFile
// method works correctly.
TEST(HyperspectralDataLoader, ReadHSIHeaderFromFile) {
  super_resolution::hyperspectral::HSIBinaryDataParameters parameters;
  parameters.ReadHeaderFromFile(kTestHeaderPath);
  EXPECT_EQ(
      parameters.data_format.interleave,
      super_resolution::hyperspectral::HSI_BINARY_INTERLEAVE_BSQ);
  EXPECT_EQ(
      parameters.data_format.data_type,
      super_resolution::hyperspectral::HSI_DATA_TYPE_FLOAT);
  EXPECT_EQ(parameters.data_format.big_endian, false);
  EXPECT_EQ(parameters.header_offset, 0);
  EXPECT_EQ(parameters.num_data_rows, 11620);
  EXPECT_EQ(parameters.num_data_cols, 11620);
  EXPECT_EQ(parameters.num_data_bands, 1506);
}

// Test reading in binary hyperspectral image data.
TEST(HyperspectralDataLoader, LoadBinaryData) {
  super_resolution::hyperspectral::HyperspectralDataLoader hs_data_loader(
      kTestConfigFilePath);
  hs_data_loader.LoadImageFromENVIFile();
  const super_resolution::ImageData image = hs_data_loader.GetImage();
  EXPECT_EQ(image.GetImageSize(), cv::Size(3, 6));
  EXPECT_EQ(image.GetNumChannels(), 5);

  // The test data values are organized as follows:
  //   The tens place (the whole value) corresponds to the band index,
  //   the first decimal corresponds to the row index, and
  //   the second decimal corresponds to the column index.
  // The range specified in the example file is bands 5-10, rows 2-8, and
  // columns 0-3. Hence, band indices start at 5, rows at 2, and columns at 0.
  const cv::Mat expected_channel_0 = (cv::Mat_<double>(6, 3)
      << 5.20, 5.21, 5.22,
         5.30, 5.31, 5.32,
         5.40, 5.41, 5.42,
         5.50, 5.51, 5.52,
         5.60, 5.61, 5.62,
         5.70, 5.71, 5.72);
  EXPECT_TRUE(AreMatricesEqual(
      image.GetChannelImage(0), expected_channel_0, kPrecisionErrorTolerance));
  const cv::Mat expected_channel_4 = (cv::Mat_<double>(6, 3)
      << 9.20, 9.21, 9.22,
         9.30, 9.31, 9.32,
         9.40, 9.41, 9.42,
         9.50, 9.51, 9.52,
         9.60, 9.61, 9.62,
         9.70, 9.71, 9.72);
  EXPECT_TRUE(AreMatricesEqual(
      image.GetChannelImage(4), expected_channel_4, kPrecisionErrorTolerance));
}

// This tests that the binary data is saved correctly by verifying that it can
// be read back without any errors in the new file.
TEST(HyperspectralDataLoader, SaveBinaryData) {
  // Read the original ENVI test file.
  super_resolution::hyperspectral::HyperspectralDataLoader hs_data_loader_1(
      kTestConfigFilePath);
  hs_data_loader_1.LoadImageFromENVIFile();
  const super_resolution::ImageData original_image =
      hs_data_loader_1.GetImage();

  // Now write the file to a temp directory, using the default supported format.
  // TODO: Once more data formats are supported, test those too.
  super_resolution::hyperspectral::HyperspectralDataLoader hs_data_loader_2(
      kTestOutputFilePath);
  super_resolution::hyperspectral::HSIBinaryDataFormat data_format;
  hs_data_loader_2.SaveImage(original_image, data_format);

  // Now try reading the file again. First, generate the config file so we can
  // read it.
  super_resolution::hyperspectral::HyperspectralDataLoader hs_data_loader_3(
      kTestOutputFilePath + ".config");  // The config file was generated.
  hs_data_loader_3.LoadImageFromENVIFile();
  const super_resolution::ImageData saved_image = hs_data_loader_3.GetImage();
  EXPECT_TRUE(AreImagesEqual(
      original_image, saved_image, kPrecisionErrorTolerance));
}
