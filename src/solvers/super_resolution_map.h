#ifndef SRC_SOLVERS_SUPER_RESOLUTION_MAP_H_
#define SRC_SOLVERS_SUPER_RESOLUTION_MAP_H_

#include <vector>

namespace super_resolution {

class MapSolver : public Solver {
 public:
  MapSolver(
      const std::vector<ImageData>& low_res_images,
      const ImageModel& image_model);

  virtual ImageData Solve();

 private:
  // here
};

}  // namespace super_resolution

#endif  // SRC_SOLVERS_SUPER_RESOLUTION_MAP_H_
