#include "util/data_loader.h"

#include <vector>

#include "image/image_data.h"

namespace super_resolution {
namespace util {

DataLoader::DataLoader(const std::string& data_path) : data_path_(data_path) {
  // TODO: implement.
}

std::vector<ImageData> DataLoader::LoadImages() const {
  // TODO: implement.
  std::vector<ImageData> images;
  return images;
}

}  // namespace util
}  // namespace super_resolution
