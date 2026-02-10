#include "kan_node.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brain19 {

KANNode::KANNode(size_t num_knots)
    : num_knots_(num_knots)
{
    if (num_knots < 4) {
        throw std::invalid_argument("KANNode requires at least 4 knots");
    }
    
    // Create uniform knot vector for cubic B-splines
    knots_.resize(num_knots_ + 4);  // +4 for cubic (degree 3)
    for (size_t i = 0; i < knots_.size(); i++) {
        knots_[i] = static_cast<double>(i) / static_cast<double>(knots_.size() - 1);
    }
    
    // Initialize coefficients to zero (identity-like)
    coefficients_.resize(num_knots_);
    std::fill(coefficients_.begin(), coefficients_.end(), 0.0);
}

double KANNode::evaluate(double x) const {
    // Clamp input to [0, 1] range
    x = std::max(0.0, std::min(1.0, x));
    
    double result = 0.0;
    for (size_t i = 0; i < coefficients_.size(); i++) {
        result += coefficients_[i] * basis_function(i, x, 3);
    }
    
    return result;
}

void KANNode::set_coefficients(const std::vector<double>& coefs) {
    if (coefs.size() != coefficients_.size()) {
        throw std::invalid_argument("Coefficient size mismatch");
    }
    coefficients_ = coefs;
}

std::vector<double> KANNode::gradient(double x, [[maybe_unused]] double epsilon) const {
    // Analytical gradient: df/dc_i = B_i(x)
    // Since f(x) = sum_i c_i * B_i(x), the gradient w.r.t. c_i is simply B_i(x)
    x = std::max(0.0, std::min(1.0, x));
    
    std::vector<double> grad(coefficients_.size());
    for (size_t i = 0; i < coefficients_.size(); i++) {
        grad[i] = basis_function(i, x, 3);
    }
    return grad;
}

double KANNode::basis_function(size_t i, double x, size_t degree) const {
    return cox_de_boor(i, degree, x);
}

double KANNode::cox_de_boor(size_t i, size_t k, double x) const {
    // Cox-de Boor recursion for B-spline basis
    if (k == 0) {
        // Fix boundary: include right endpoint for last interval
        if (i + 1 == knots_.size() - 1) {
            return (x >= knots_[i] && x <= knots_[i + 1]) ? 1.0 : 0.0;
        }
        return (x >= knots_[i] && x < knots_[i + 1]) ? 1.0 : 0.0;
    }
    
    double denom1 = knots_[i + k] - knots_[i];
    double denom2 = knots_[i + k + 1] - knots_[i + 1];
    
    double term1 = 0.0;
    if (denom1 > 1e-10) {
        term1 = (x - knots_[i]) / denom1 * cox_de_boor(i, k - 1, x);
    }
    
    double term2 = 0.0;
    if (denom2 > 1e-10) {
        term2 = (knots_[i + k + 1] - x) / denom2 * cox_de_boor(i + 1, k - 1, x);
    }
    
    return term1 + term2;
}

} // namespace brain19
