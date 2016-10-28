// The ImageModel class specifies a forward image formation model that can
// produce a LR image given an HR image. It is defined as a series of
// degradations (DegradationOperators) that downsample, blur, move, etc. the
// original HR image to simulate an observation.
//
// The model can be used to either generate simulated image data, or as a
// component in some objective functions, such as that used in MAP.

#ifndef SRC_IMAGE_MODEL_IMAGE_MODEL_H_
#define SRC_IMAGE_MODEL_IMAGE_MODEL_H_

#include <memory>
#include <vector>

#include "image/image_data.h"
#include "image_model/degradation_operator.h"

namespace super_resolution {

class ImageModel {
 public:
  // Adds a DegradationOperator to the model. These operators should be added in
  // the order that they need to be applied. For example, if the image model is
  //    Y = DBMX + n
  // where Y and X are the LR and HR images, respectively, then add order
  // should be M, B, D, n.
  // Note that additive operators come last due to the order of operations.
  void AddDegradationOperator(
      std::unique_ptr<DegradationOperator> degradation_operator);

  // Apply this forward model to the given image at the given index in the
  // multiframe sequence.
  void ApplyModel(ImageData* image_data, const int index) const;

 private:
  // An ordered list of degradation operators, to be applied in this order. We
  // keep pointers because the DegradationOperator class is pure virtual.
  std::vector<std::unique_ptr<DegradationOperator>> degradation_operators_;
};

}  // namespace super_resolution

#endif  // SRC_IMAGE_MODEL_IMAGE_MODEL_H_
