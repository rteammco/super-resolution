// Provides an API to load and store FT-IR data sets.

#ifndef SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_
#define SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_

#include <string>
#include <vector>

#include "image/image_data.h"

namespace super_resolution {
namespace hyperspectral {

class HyperspectralDataLoader {
 public:
  // Provide a data file name that will be processed or written out to. If the
  // data is being read from a file, the file should contain meta information
  // about the image size and number of spectral bands.
  explicit HyperspectralDataLoader(const std::string& file_path)
      : file_path_(file_path) {}

  // Call this to actually execute the data load process using the information
  // provided to the constructor. If the data load process was unsuccessful or
  // if the data file size does not matche the given data_size value, an error
  // check will fail.
  void LoadData();

  // Returns the ImageData object containing the hyperspectral image data. The
  // image will be empty if LoadData() was not called.
  ImageData GetImage() const;

  // Writes the given image to the data path. This will not store the image for
  // the GetImage() method.
  //
  // TODO: Allow specifying the type of file to be generated (e.g. binary or
  // text).
  void WriteImage(const ImageData& image) const;

 private:
  // The name of the data file to be loaded.
  const std::string& file_path_;

  // The data is stored in an ImageData container.
  ImageData hyperspectral_image_;
};

}  // namespace hyperspectral
}  // namespace super_resolution

#endif  // SRC_HYPERSPECTRAL_HYPERSPECTRAL_DATA_LOADER_H_
