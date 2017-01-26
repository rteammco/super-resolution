// Provides an API to load and store FT-IR data sets.

#ifndef SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_
#define SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_

#include <string>
#include <vector>

#include "image/image_data.h"

namespace super_resolution {
namespace hyperspectral {

// Dimensions of the Hyperspectral cube.
struct HyperspectralCubeSize {
  HyperspectralCubeSize(const int rows, const int cols, const int bands)
      : rows(rows), cols(cols), bands(bands) {}

  const int rows;
  const int cols;
  const int bands;
};

class HyperspectralDataLoader {
 public:
  // Provide a data file name that will be processed.
  HyperspectralDataLoader(
      const std::string& file_path, const HyperspectralCubeSize& data_size);

  // Call this to actually execute the data load process using the information
  // provided to the constructor. If the data load process was unsuccessful or
  // if the data file size does not matche the given data_size value, an error
  // check will fail.
  void LoadData();

  // Returns the ImageData object containing the hyperspectral image data. The
  // image will be empty if LoadData() was not called.
  const ImageData& GetImage() const;

 private:
  // The name of the data file to be loaded.
  const std::string& file_path_;

  // The size of the hyperspectral data cube. This must be known to correctly
  // parse the hyperspectral data file.
  const HyperspectralCubeSize data_size_;

  // The data is stored in an ImageData container.
  ImageData hyperspectral_image_;
};

}  // namespace hyperspectral
}  // namespace super_resolution

#endif  // SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_
