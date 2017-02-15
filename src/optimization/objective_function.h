// Defines a common ObjectiveFunction class that can be customized with
// different objective terms as needed. The point of this class is to provide a
// generic interface for defining a cost function that can be minimized by an
// optimizer. All solver strategies can define their own objective functions as
// needed and pass it to the optimizer.

#ifndef SRC_OPTIMIZATION_OBJECTIVE_FUNCTION_H_
#define SRC_OPTIMIZATION_OBJECTIVE_FUNCTION_H_

#include <memory>
#include <vector>

namespace super_resolution {

// An ObjectiveTerm computes the cost and gradient of a part of the objective
// function given variables needed to do those computations. For example, a
// term may be the data fidelity term, or one of several regularization terms.
class ObjectiveTerm {
 public:
  // Compute the cost and gradient of this objective term. Each term must be
  // implemented as needed for that specific computation.
  //
  // NOTE: The gradient may be NULL (nullptr), in which case it is not computed.
  virtual double Compute(
      const double* estimated_image_data, double* gradient) const = 0;
};

// The ObjectiveFunction is just a collection of ObjectiveTerms which are
// computed independently.
class ObjectiveFunction {
 public:
  explicit ObjectiveFunction(const int num_parameters)
      : num_parameters_(num_parameters) {}

  // Add a new ObjectiveTerm to the list.
  void AddTerm(const std::shared_ptr<ObjectiveTerm> objective_term) {
    terms_.push_back(objective_term);
  }

  // Computes all terms and returns the sum of the residual costs and the sum
  // of the gradients. If gradient is NULL, it will not be computed.
  double ComputeAllTerms(
      const double* estimated_image_data, double* gradient = nullptr) const;

 private:
  // The number of parameters in the given estimated_image_data. This is also
  // the number of variables in the gradient vector.
  const int num_parameters_;

  std::vector<std::shared_ptr<ObjectiveTerm>> terms_;
};

}  // namespace super_resolution

#endif  // SRC_OPTIMIZATION_OBJECTIVE_FUNCTION_H_
