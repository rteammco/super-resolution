#include "hyperspectral/hyperspectral_data_loader.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "image/image_data.h"
#include "util/config_reader.h"
#include "util/matrix_util.h"

#include "opencv2/core/core.hpp"

#include "glog/logging.h"

namespace super_resolution {
namespace {

constexpr char kMatlabTextDataDelimiter = ',';

struct HSIDataRange {
  int start_row = 0;
  int start_col = 0;
  int end_row = 0;
  int end_col = 0;
  int start_band = 0;
  int end_band = 0;
};


// Reverses the bytes of the given value (e.g. float). This is used to convert
// from the data's endian form into the machine's endian form when they are not
// matched up.
template <typename T>
T ReverseBytes(const T value) {
  T reversed_value;
  const unsigned char* original_bytes = (const unsigned char*)(&value);
  unsigned char* reversed_bytes  = (unsigned char*)(&reversed_value);
  const int num_bytes = sizeof(T);
  for (int i = 0; i < num_bytes; ++i) {
    reversed_bytes[i] = original_bytes[num_bytes - 1 - i];
  }
  return reversed_value;
}

// Returns true if this machine uses big-endian or false if it uses
// little-endian. If the byte order is mismatched with the data's byte order,
// the bytes will have to be reveresed.
bool IsMachineBigEndian() {
  // Determine the machine endian. Union of memory means "unsigned int value"
  // and "unsigned char bytes" share the same memory.
  union UnsignedNumber {
    unsigned int value;
    unsigned char bytes[sizeof(unsigned int)];
  };
  // Set the value to unsigned 1, and check the byte array to see if it is in
  // big endian or little endian order. The left-most byte will be empty (zero)
  // if the machine is big endian.
  UnsignedNumber number;
  number.value = 1U;  // Unsigned int 1.
  const bool machine_big_endian = (number.bytes[0] != 1U);
  return machine_big_endian;
}

template <typename T>
ImageData ReadBinaryFileBSQ(
    const std::string& hsi_file_path,
    const int num_data_rows,
    const int num_data_cols,
    const int num_data_bands,
    const long start_index,
    const bool reverse_bytes,
    const HSIDataRange& data_range) {

  std::ifstream input_file(hsi_file_path);
  CHECK(input_file.is_open())
      << "File '" << hsi_file_path << "' could not be opened for reading.";

  const int data_point_size = sizeof(T);
  long current_index = start_index;
  input_file.seekg(current_index * data_point_size);

  ImageData hsi_image;
  const cv::Size image_size(
      data_range.end_col - data_range.start_col,
      data_range.end_row - data_range.start_row);
  const long num_pixels = num_data_rows * num_data_cols;
  for (int band = data_range.start_band; band < data_range.end_band; ++band) {
    const long band_index = band * num_pixels;
    cv::Mat channel_image(image_size, util::kOpenCvMatrixType);
    for (int row = data_range.start_row; row < data_range.end_row; ++row) {
      const int channel_row = row - data_range.start_row;
      for (int col = data_range.start_col; col < data_range.end_col; ++col) {
        const int channel_col = col - data_range.start_col;
        const long pixel_index = row * num_data_cols + col;
        const long next_index = band_index + pixel_index;
        // Skip to next position if necessary.
        if (next_index > (current_index + 1)) {
          input_file.seekg(next_index * data_point_size);
        }
        T value;
        input_file.read(reinterpret_cast<char*>(&value), data_point_size);
        if (reverse_bytes) {
          value = ReverseBytes<T>(value);
        }
        channel_image.at<double>(channel_row, channel_col) =
            static_cast<double>(value);
        current_index = next_index;
      }
    }
    hsi_image.AddChannel(channel_image, DO_NOT_NORMALIZE_IMAGE);
  }
  input_file.close();
  return hsi_image;
}

template <typename T>
void WriteBinaryFileBSQ(
    const ImageData& image,
    const std::string& hsi_file_path,
    const bool reverse_bytes) {

  // Write the binary file.
  std::ofstream output_envi_file(hsi_file_path);
  CHECK(output_envi_file.is_open())
      << "ENVI file '" << hsi_file_path << "' could not be opened for writing.";
  const cv::Size image_size = image.GetImageSize();
  const int num_rows = image_size.height;
  const int num_cols = image_size.width;
  const int num_bands = image.GetNumChannels();
  const int data_point_size = sizeof(T);
  for (int band = 0; band < num_bands; ++band) {
    for (int row = 0; row < num_rows; ++row) {
      for (int col = 0; col < num_cols; ++col) {
        const double pixel_value = image.GetPixelValue(band, row, col);
        T output_value = static_cast<T>(pixel_value);
        if (reverse_bytes) {
          output_value = ReverseBytes<T>(output_value);
        }
        output_envi_file.write(
            reinterpret_cast<char*>(&output_value), data_point_size);
      }
    }
  }
  output_envi_file.close();

  // Write the header file.
  const std::string header_file_path = hsi_file_path + ".hdr";
  std::ofstream output_header_file(header_file_path);
  CHECK(output_header_file.is_open())
      << "Header file '" << header_file_path
      << "' could not be opened for writing.";
  output_header_file << "ENVI\n";
  output_header_file
      << "description = {File generated by HyperspectralDataLoader.}\n";
  output_header_file << "samples = " << num_rows << "\n";
  output_header_file << "lines = " << num_cols << "\n";
  output_header_file << "bands = " << num_bands << "\n";
  output_header_file << "header offset = 0\n";
  output_header_file << "file type = ENVI Standard\n";
  output_header_file << "data type = 4\n";  // TODO: 4 = float, might change.
  output_header_file << "interleave = bsq\n";
  output_header_file << "byte order = 0\n";  // TODO: This might also change.
  // TODO: Verify that we don't need to generate the other "unknown" options.
  output_header_file.close();

  // Also write the config file so we can easily read it again later.
  const std::string config_file_path = hsi_file_path + ".config";
  std::ofstream output_config_file(config_file_path);
  CHECK(output_config_file.is_open())
      << "Configuration file '" << config_file_path
      << "' could not be opened for writing.";
  output_config_file
      << "# Configuration file for reading '" << hsi_file_path
      << "', generated by HyperspectralDataLoader.\n";
  output_config_file << "file " << hsi_file_path << "\n";
  output_config_file << "interleave bsq\n";
  output_config_file << "data_type float\n";  // TODO: Might not be float.
  output_config_file << "big_endian false\n";  // TODO: Might not be false.
  output_config_file << "header_offset 0\n";
  output_config_file << "num_data_rows " << num_rows << "\n";
  output_config_file << "num_data_cols " << num_cols << "\n";
  output_config_file << "num_data_bands " << num_bands << "\n";
  output_config_file << "start_row " << 0 << "\n";
  output_config_file << "end_row " << num_rows << "\n";
  output_config_file << "start_col " << 0 << "\n";
  output_config_file << "end_col " << num_cols << "\n";
  output_config_file << "start_band " << 0 << "\n";
  output_config_file << "end_band " << num_bands << "\n";
  output_config_file.close();
}

ImageData ReadBinaryFile(
    const std::string& hsi_file_path,
    const HSIBinaryDataParameters& parameters,
    const HSIDataRange& data_range) {

  // If endians don't match, the bytes from the file have to be reversed.
  const bool machine_big_endian = IsMachineBigEndian();
  const bool reverse_bytes =
      (parameters.data_format.big_endian != machine_big_endian);

  // TODO: This may change, depending on interleave format and data type.
  return ReadBinaryFileBSQ<float>(
      hsi_file_path,
      parameters.num_data_rows,
      parameters.num_data_cols,
      parameters.num_data_bands,
      parameters.header_offset,
      reverse_bytes,
      data_range);
}

}  // namespace

void HSIBinaryDataParameters::ReadHeaderFromFile(
    const std::string& header_file_path) {

  util::ConfigurationFileReader config_reader;
  config_reader.SetDelimiter('=');
  config_reader.ReadFromFile(header_file_path);
  if (config_reader.HasValue("interleave")) {
    const std::string interleave = config_reader.GetValue("interleave");
    if (interleave == "bsq") {
      data_format.interleave = HSI_BINARY_INTERLEAVE_BSQ;
    } else {
      LOG(WARNING) << "Unknown/unsupported interleave format: "
                   << interleave << ". Using BSQ by default.";
    }
  }
  if (config_reader.HasValue("data type")) {
    const std::string data_type = config_reader.GetValue("data type");
    if (data_type == "4") {
      data_format.data_type = HSI_DATA_TYPE_FLOAT;
    } else {
      LOG(WARNING) << "Unknown/unsupported data type: "
                   << data_type << ". Using float by default.";
    }
  }
  if (config_reader.HasValue("byte order")) {
    const std::string byte_order = config_reader.GetValue("byte order");
    if (byte_order == "1") {
      data_format.big_endian = true;
    } else {
      data_format.big_endian = false;
    }
  }
  if (config_reader.HasValue("header offset")) {
    header_offset = config_reader.GetValueAsInt("header offset");
  }
  if (config_reader.HasValue("samples")) {
    num_data_rows = config_reader.GetValueAsInt("samples");
  }
  if (config_reader.HasValue("lines")) {
    num_data_cols = config_reader.GetValueAsInt("lines");
  }
  if (config_reader.HasValue("bands")) {
    num_data_bands = config_reader.GetValueAsInt("bands");
  }
}

// TODO: Support for different interleaves and data types.
// TODO: Allow a header to take place of some of the config file values (i.e.
//       data size and format parameters) if the "header" key is given. Right
//       now config file has to contain all of the information directly.
void HyperspectralDataLoader::LoadImageFromENVIFile() {
  util::ConfigurationFileReader config_reader;
  config_reader.SetDelimiter(' ');
  config_reader.ReadFromFile(file_path_);

  // Get the path of the binary data file.
  const std::string hsi_file_path = config_reader.GetValueOrDie("file");

  // Get all of the necessary HSI file metadata.
  HSIBinaryDataParameters parameters;
  // Interleave format:
  const std::string interleave = config_reader.GetValueOrDie("interleave");
  if (interleave == "bsq") {
    parameters.data_format.interleave = HSI_BINARY_INTERLEAVE_BSQ;
  } else {
    LOG(FATAL) << "Unsupported interleave format: '" << interleave << "'.";
  }
  // Data type:
  const std::string data_type = config_reader.GetValueOrDie("data_type");
  if (data_type == "float") {
    parameters.data_format.data_type = HSI_DATA_TYPE_FLOAT;
  } else {
    LOG(FATAL) << "Unsupported data type: '" << data_type << "'.";
  }
  // Endian:
  const std::string big_endian = config_reader.GetValueOrDie("big_endian");
  if (big_endian == "true") {
    parameters.data_format.big_endian = true;
  } else {
    parameters.data_format.big_endian = false;
  }
  // Header offset:
  const std::string header_offset =
      config_reader.GetValueOrDie("header_offset");
  parameters.header_offset = std::atoi(header_offset.c_str());
  CHECK_GE(parameters.header_offset, 0)
      << "Header offset must be non-negative.";
  // Number of rows:
  const std::string num_data_rows =
      config_reader.GetValueOrDie("num_data_rows");
  parameters.num_data_rows = std::atoi(num_data_rows.c_str());
  CHECK_GT(parameters.num_data_rows, 0)
      << "Number of data rows must be positive.";
  // Number of columns:
  const std::string num_data_cols =
      config_reader.GetValueOrDie("num_data_cols");
  parameters.num_data_cols = std::atoi(num_data_cols.c_str());
  CHECK_GT(parameters.num_data_cols, 0)
      << "Number of data cols must be positive.";
  // Number of spectral bands:
  const std::string num_data_bands =
      config_reader.GetValueOrDie("num_data_bands");
  parameters.num_data_bands = std::atoi(num_data_bands.c_str());
  CHECK_GT(parameters.num_data_bands, 0)
      << "Number of data bands must be positive.";

  // Now get the data range parameters.
  HSIDataRange data_range;
  // Start row:
  const std::string start_row_string = config_reader.GetValueOrDie("start_row");
  data_range.start_row = std::atoi(start_row_string.c_str());
  CHECK_GE(data_range.start_row, 0) << "Start row index cannot be negative.";
  CHECK_LT(data_range.start_row, parameters.num_data_rows)
      << "Start row index is out of bounds.";
  // End row:
  const std::string end_row_string = config_reader.GetValueOrDie("end_row");
  data_range.end_row = std::atoi(end_row_string.c_str());
  CHECK_GT(data_range.end_row, 0) << "End row index must be positive.";
  CHECK_LE(data_range.end_row, parameters.num_data_rows)
      << "End row index is out of bounds.";
  CHECK_GT(data_range.end_row - data_range.start_row, 0)
      << "Row range must be positive.";
  // Start column:
  const std::string start_col_string = config_reader.GetValueOrDie("start_col");
  data_range.start_col = std::atoi(start_col_string.c_str());
  CHECK_GE(data_range.start_col, 0) << "Start column index cannot be negative.";
  CHECK_LT(data_range.start_col, parameters.num_data_cols)
      << "Start column index is out of bounds.";
  // End column:
  const std::string end_col_string = config_reader.GetValueOrDie("end_col");
  data_range.end_col = std::atoi(end_col_string.c_str());
  CHECK_GT(data_range.end_col, 0) << "End column index must be positive.";
  CHECK_LE(data_range.end_col, parameters.num_data_cols)
      << "End column index is out of bounds.";
  CHECK_GT(data_range.end_col - data_range.start_col, 0)
      << "Column range must be positive.";
  // Start band:
  const std::string start_band_string =
      config_reader.GetValueOrDie("start_band");
  data_range.start_band = std::atoi(start_band_string.c_str());
  CHECK_GE(data_range.start_band, 0) << "Start band index cannot be negative.";
  CHECK_LT(data_range.start_band, parameters.num_data_bands)
      << "Start band index is out of bounds.";
  // End band:
  const std::string end_band_string = config_reader.GetValueOrDie("end_band");
  data_range.end_band = std::atoi(end_band_string.c_str());
  CHECK_GT(data_range.end_band, 0) << "End band index must be positive.";
  CHECK_LE(data_range.end_band, parameters.num_data_bands)
      << "End band index is out of bounds.";
  CHECK_GT(data_range.end_band - data_range.start_band, 0)
      << "Band range must be positive.";

  // And finally, read the data according to the parameters and range.
  hyperspectral_image_ = ReadBinaryFile(hsi_file_path, parameters, data_range);
}

ImageData HyperspectralDataLoader::GetImage() const {
  CHECK_GT(hyperspectral_image_.GetNumChannels(), 0)
      << "The hyperspectral image is empty. "
      << "Make sure to call LoadData() first.";
  return hyperspectral_image_;
}

void HyperspectralDataLoader::SaveImage(
    const ImageData& image,
    const HSIBinaryDataFormat& binary_data_format) const {

  // If endians don't match, the bytes from the file have to be reversed.
  const bool machine_big_endian = IsMachineBigEndian();
  const bool reverse_bytes =
      (binary_data_format.big_endian != machine_big_endian);

  // TODO: This may change, depending on interleave format and data type.
  WriteBinaryFileBSQ<float>(image, file_path_, reverse_bytes);
}

}  // namespace super_resolution
