#include "kan_module.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brain19 {

// Multi-layer constructor
KANModule::KANModule(const std::vector<size_t>& layer_dims, size_t num_knots)
    : layer_dims_(layer_dims)
{
    if (layer_dims.size() < 2) {
        throw std::invalid_argument("KANModule topology requires at least 2 dimensions (input + output)");
    }
    for (auto d : layer_dims) {
        if (d == 0) throw std::invalid_argument("KANModule requires non-zero dimensions");
    }
    
    input_dim_ = layer_dims.front();
    output_dim_ = layer_dims.back();
    
    layers_.reserve(layer_dims.size() - 1);
    for (size_t k = 0; k + 1 < layer_dims.size(); k++) {
        layers_.push_back(std::make_unique<KANLayer>(layer_dims[k], layer_dims[k + 1], num_knots));
    }
}

// Legacy single-layer constructor (backward compatible)
KANModule::KANModule(size_t input_dim, size_t output_dim, size_t num_knots)
    : input_dim_(input_dim)
    , output_dim_(output_dim)
    , layer_dims_({input_dim, output_dim})
{
    if (input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument("KANModule requires non-zero dimensions");
    }
    
    // Single layer: input_dim → output_dim
    layers_.push_back(std::make_unique<KANLayer>(input_dim, output_dim, num_knots));
}

std::shared_ptr<KANModule> KANModule::clone() const {
    // Determine num_knots from first layer's first node
    size_t num_knots = 10;  // default
    if (!layers_.empty() && layers_[0]->num_nodes() > 0) {
        num_knots = layers_[0]->node(0, 0).get_coefficients().size();
    }

    auto copy = std::make_shared<KANModule>(layer_dims_, num_knots);

    // Deep copy all B-spline coefficients
    for (size_t k = 0; k < layers_.size(); ++k) {
        const auto& src_nodes = layers_[k]->get_nodes();
        auto& dst_nodes = copy->layers_[k]->get_nodes_mutable();
        for (size_t n = 0; n < src_nodes.size(); ++n) {
            dst_nodes[n]->set_coefficients(src_nodes[n]->get_coefficients());
        }
    }

    return copy;
}

std::vector<std::vector<double>> KANModule::forward_all(const std::vector<double>& inputs) const {
    // Returns activations[0] = inputs, activations[k+1] = output of layer k
    std::vector<std::vector<double>> activations;
    activations.reserve(layers_.size() + 1);
    activations.push_back(inputs);
    
    for (size_t k = 0; k < layers_.size(); k++) {
        activations.push_back(layers_[k]->evaluate(activations[k]));
    }
    
    return activations;
}

std::vector<double> KANModule::evaluate(const std::vector<double>& inputs) const {
    if (inputs.size() != input_dim_) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    std::vector<double> current = inputs;
    for (size_t k = 0; k < layers_.size(); k++) {
        current = layers_[k]->evaluate(current);
    }
    
    return current;
}

KanTrainingResult KANModule::train(
    const std::vector<DataPoint>& dataset,
    const KanTrainingConfig& config
) {
    if (dataset.empty()) {
        throw std::invalid_argument("Training dataset is empty");
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    KanTrainingResult result;
    result.iterations_run = 0;
    result.converged = false;
    
    double prev_loss = compute_loss(dataset);
    
    for (size_t iter = 0; iter < config.max_iterations; iter++) {
        gradient_descent_step(dataset, config.learning_rate);
        
        double current_loss = compute_loss(dataset);
        result.iterations_run = iter + 1;
        result.final_loss = current_loss;
        
        // Check convergence
        if (std::abs(prev_loss - current_loss) < config.convergence_threshold) {
            result.converged = true;
            break;
        }
        
        prev_loss = current_loss;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    
    return result;
}

double KANModule::compute_mse(const std::vector<DataPoint>& dataset) const {
    return compute_loss(dataset);
}

double KANModule::compute_loss(const std::vector<DataPoint>& dataset) const {
    double total_error = 0.0;
    size_t count = 0;
    
    for (const auto& point : dataset) {
        auto predictions = evaluate(point.inputs);
        
        for (size_t i = 0; i < output_dim_; i++) {
            double error = predictions[i] - point.outputs[i];
            total_error += error * error;
            count++;
        }
    }
    
    return count > 0 ? total_error / static_cast<double>(count) : 0.0;
}

void KANModule::gradient_descent_step(
    const std::vector<DataPoint>& dataset,
    double learning_rate
) {
    // For multi-layer KAN, we use layer-wise gradient computation.
    // Since f(x) = sum_i c_i * B_i(x), gradient of each node output 
    // w.r.t. its coefficients is analytical: ∂φ/∂c_i = B_i(x).
    //
    // For multi-layer, we need chain rule. But since KAN layers are 
    // sums of univariate functions, the gradient of output j of layer k
    // w.r.t. coefficient c of node(i,j) is simply B_c(activation_i).
    //
    // The chain rule through layers: we compute ∂L/∂activation for each layer
    // and propagate backward.
    
    for (const auto& point : dataset) {
        // Forward pass: collect all activations
        auto activations = forward_all(point.inputs);
        
        // Final output
        const auto& predictions = activations.back();
        
        // Compute output error gradient: dL/d_output_j = 2*(pred_j - target_j) / N
        std::vector<double> d_output(output_dim_);
        for (size_t j = 0; j < output_dim_; j++) {
            d_output[j] = 2.0 * (predictions[j] - point.outputs[j]) / static_cast<double>(dataset.size());
        }
        
        // Backward pass through layers
        std::vector<double> d_activation = d_output;  // gradient w.r.t. current layer output
        
        for (int k = static_cast<int>(layers_.size()) - 1; k >= 0; k--) {
            auto& layer = *layers_[k];
            const auto& input_act = activations[k];  // input to this layer
            size_t n_in = layer.input_dim();
            size_t n_out = layer.output_dim();
            
            // Gradient w.r.t. input of this layer (for propagation to previous layer)
            std::vector<double> d_input(n_in, 0.0);
            
            for (size_t i = 0; i < n_in; i++) {
                for (size_t j = 0; j < n_out; j++) {
                    auto& nd = layer.node(i, j);
                    
                    // Update coefficients of node(i,j)
                    // ∂L/∂c = d_activation[j] * B_c(input_act[i])
                    auto node_grad = nd.gradient(input_act[i]);
                    auto coefs = nd.get_coefficients();
                    
                    for (size_t c = 0; c < coefs.size(); c++) {
                        coefs[c] -= learning_rate * d_activation[j] * node_grad[c];
                    }
                    nd.set_coefficients(coefs);
                    
                    // For backprop: ∂L/∂input_i += d_activation[j] * ∂φ_{i,j}/∂x_i
                    // ∂φ/∂x = sum_c c_c * ∂B_c/∂x (numerical)
                    // Approximate with finite differences
                    if (k > 0) {
                        double eps = 1e-6;
                        double x = input_act[i];
                        double f_plus = nd.evaluate(std::min(1.0, x + eps));
                        double f_minus = nd.evaluate(std::max(0.0, x - eps));
                        double df_dx = (f_plus - f_minus) / (2.0 * eps);
                        d_input[i] += d_activation[j] * df_dx;
                    }
                }
            }
            
            d_activation = d_input;
        }
    }
}

} // namespace brain19
