#include "kan_layer.hpp"
#include <stdexcept>

namespace brain19 {

KANLayer::KANLayer(size_t input_dim, size_t num_knots_per_node) {
    if (input_dim == 0) {
        throw std::invalid_argument("KANLayer requires at least 1 input");
    }
    
    nodes_.reserve(input_dim);
    for (size_t i = 0; i < input_dim; i++) {
        nodes_.push_back(std::make_unique<KANNode>(num_knots_per_node));
    }
}

std::vector<double> KANLayer::evaluate(const std::vector<double>& inputs) const {
    if (inputs.size() != nodes_.size()) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    std::vector<double> outputs(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); i++) {
        outputs[i] = nodes_[i]->evaluate(inputs[i]);
    }
    
    return outputs;
}

} // namespace brain19
