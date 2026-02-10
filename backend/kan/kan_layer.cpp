#include "kan_layer.hpp"
#include <stdexcept>

namespace brain19 {

KANLayer::KANLayer(size_t input_dim, size_t output_dim, size_t num_knots_per_node)
    : input_dim_(input_dim)
    , output_dim_(output_dim)
{
    if (input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument("KANLayer requires non-zero dimensions");
    }
    
    nodes_.reserve(input_dim * output_dim);
    for (size_t i = 0; i < input_dim * output_dim; i++) {
        nodes_.push_back(std::make_unique<KANNode>(num_knots_per_node));
    }
}

std::vector<double> KANLayer::evaluate(const std::vector<double>& inputs) const {
    if (inputs.size() != input_dim_) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    std::vector<double> outputs(output_dim_, 0.0);
    
    for (size_t j = 0; j < output_dim_; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < input_dim_; i++) {
            sum += nodes_[i * output_dim_ + j]->evaluate(inputs[i]);
        }
        outputs[j] = sum;
    }
    
    return outputs;
}

KANNode& KANLayer::node(size_t i, size_t j) {
    if (i >= input_dim_ || j >= output_dim_) {
        throw std::out_of_range("KANLayer::node index out of range");
    }
    return *nodes_[i * output_dim_ + j];
}

const KANNode& KANLayer::node(size_t i, size_t j) const {
    if (i >= input_dim_ || j >= output_dim_) {
        throw std::out_of_range("KANLayer::node index out of range");
    }
    return *nodes_[i * output_dim_ + j];
}

} // namespace brain19
