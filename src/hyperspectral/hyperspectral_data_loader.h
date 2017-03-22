// Provides an API to load and store FT-IR data sets.

#ifndef SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_
#define SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_

#include <string>
#include <vector>

#include "image/image_data.h"

namespace super_resolution {
namespace hyperspectral {

// The possible formats of the hyperspectral image data to be loaded. Binary
// formats do not specify any information other than the data itself, so header
// information must be provided separately.
//
// TODO: Add support for BIP and BIL binary formats.
enum HSIDataInterleaveFormat {
  // BSQ (band sequential) is a binary data format organized in order of
  // bands(rows(cols)). For example, for a file with 2 bands, 2 rows, and 2
  // columns, the order would be as follows:
  //   b0,r0,c0
  //   b0,r0,c1
  //   b0,r1,c0
  //   b0,r1,c1
  //   b1,r0,c0
  //   b1,r0,c1
  //   b1,r1,c0
  //   b1,r1,c1
  HSI_BINARY_INTERLEAVE_BSQ
};

// The data type dictates how the binary HSI data is stored (e.g. as doubles,
// floats, unsigned ints, etc.).
//
// TODO: Add support for other data types.
enum HSIBinaryDataType {
  HSI_DATA_TYPE_FLOAT
};

// Defines the formatting of the binary data file. This is used for reading and
// writing binary files in the appropriate way.
struct HSIBinaryDataFormat {
  HSIDataInterleaveFormat interleave = HSI_BINARY_INTERLEAVE_BSQ;
  HSIBinaryDataType data_type = HSI_DATA_TYPE_FLOAT;
  bool big_endian = false;
};

// Specifies parameters for reading binary HSI data. This information can
// either be specified manually (or through a configuration file), or by
// reading a header file provided with the HSI binary data.
struct HSIBinaryDataParameters {
  // Default constructor so this object can be made const.
  HSIBinaryDataParameters() {}

  // Attempts to read the header information from an HSI header file. If the
  // given header file path is invalid or the read fails for some reason, an
  // error will occur.
  // TODO: Implement.
  void ReadHeaderFromFile(const std::string& header_file_path);

  // The format and type of the data.
  HSIBinaryDataFormat data_format;

  // Offset of the header (if there is a header directly attached to the data).
  int header_offset = 0;

  // The size of the data. This is NOT the size of the chunk of data you want
  // to read, but rather of the entire data, even if you don't read everything.
  // These must all be non-zero.
  int num_data_rows = 0;
  int num_data_cols = 0;
  int num_data_bands = 0;
};

class HyperspectralDataLoader {
 public:
  // The given file path can serve two potential purposes:
  //
  // 1. If the data is to be read from a binary hyperspectral file (ENVI), then
  //    file_path should be the path to a CONFIGURATION file containing all
  //    information about the HSI data file.
  //
  // 2. If the data is to be written to a file, then the given file_path will
  //    be the produced output file. A header file (with an appended .hdr to
  //    the given file_path) will also be generated.
  explicit HyperspectralDataLoader(const std::string& file_path)
      : file_path_(file_path) {}

  // Attempts to load binary data. This assumes the file given to the
  // constructor is a configuration file which specifies all the necessary data
  // parameters.
  void LoadImageFromENVIFile();

  // Returns the ImageData object containing the hyperspectral image data. The
  // image will be empty if one of the LoadData methods was never called.
  ImageData GetImage() const;

  // Saves the image as a binary ENVI hyperspectral data file. The image will
  // be saved to file_path_ as given in the constructor. This will also
  // generate a .hdr (header) file in the same directory that will contain
  // header information about the data (i.e. data formatting, data size, etc.).
  //
  // The file formatting is dictated by the given HSIBinaryDataFormat.
  void SaveImage(
      const ImageData& image,
      const HSIBinaryDataFormat& binary_data_format) const;

 private:
  // The name of the data file to be loaded.
  const std::string& file_path_;

  // The data is stored in an ImageData container.
  ImageData hyperspectral_image_;
};

}  // namespace hyperspectral
}  // namespace super_resolution

#endif  // SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_
