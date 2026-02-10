#pragma once

#include "kan_node.hpp"
#include <vector>
#include <memory>

namespace brain19 {

// KANLayer: n_in × n_out grid of KANNodes
// Each edge (i,j) has its own learnable spline function
// Output j = sum_i phi_{i,j}(input_i)
class KANLayer {
public:
    // Constructor: input_dim × output_dim nodes
    KANLayer(size_t input_dim, size_t output_dim, size_t num_knots_per_node = 10);
    
    // Evaluate layer: returns vector of size output_dim
    std::vector<double> evaluate(const std::vector<double>& inputs) const;
    
    // Access node at edge (i, j) where i=input_idx, j=output_idx
    KANNode& node(size_t i, size_t j);
    const KANNode& node(size_t i, size_t j) const;
    
    // Legacy access (flat list)
    const std::vector<std::unique_ptr<KANNode>>& get_nodes() const { return nodes_; }
    std::vector<std::unique_ptr<KANNode>>& get_nodes_mutable() { return nodes_; }
    
    size_t input_dim() const { return input_dim_; }
    size_t output_dim() const { return output_dim_; }
    size_t num_nodes() const { return nodes_.size(); }
    
private:
    size_t input_dim_;
    size_t output_dim_;
    // nodes_ stored in row-major: nodes_[i * output_dim_ + j] = edge (i, j)
    std::vector<std::unique_ptr<KANNode>> nodes_;
};

} // namespace brain19
