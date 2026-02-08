#pragma once

#include "kan_node.hpp"
#include <vector>
#include <memory>

namespace brain19 {

// KANLayer: Collection of KANNodes
// Performs additive combination only (no nonlinear mixing)
class KANLayer {
public:
    // Constructor: creates input_dim nodes
    explicit KANLayer(size_t input_dim, size_t num_knots_per_node = 10);
    
    // Evaluate layer: returns vector of node outputs
    std::vector<double> evaluate(const std::vector<double>& inputs) const;
    
    // Access nodes (for training)
    const std::vector<std::unique_ptr<KANNode>>& get_nodes() const { return nodes_; }
    std::vector<std::unique_ptr<KANNode>>& get_nodes_mutable() { return nodes_; }
    
    size_t input_dim() const { return nodes_.size(); }
    
private:
    std::vector<std::unique_ptr<KANNode>> nodes_;
};

} // namespace brain19
