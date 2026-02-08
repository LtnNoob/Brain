#pragma once

#include <vector>
#include <cstddef>

namespace brain19 {

// KANNode: Univariate learnable function
// Uses B-spline representation for smooth approximation
// Input: single double, Output: single double
class KANNode {
public:
    // Constructor: num_knots defines spline resolution
    explicit KANNode(size_t num_knots = 10);
    
    // Evaluate function at given input
    double evaluate(double x) const;
    
    // Get/set spline coefficients (for training)
    const std::vector<double>& get_coefficients() const { return coefficients_; }
    void set_coefficients(const std::vector<double>& coefs);
    
    // Numerical gradient of output w.r.t. coefficients
    std::vector<double> gradient(double x, double epsilon = 1e-6) const;
    
private:
    size_t num_knots_;
    std::vector<double> knots_;         // Uniform knot vector
    std::vector<double> coefficients_;  // Spline coefficients
    
    // B-spline basis functions (cubic)
    double basis_function(size_t i, double x, size_t degree) const;
    double cox_de_boor(size_t i, size_t k, double x) const;
};

} // namespace brain19
