#include "kan_module.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brain19 {

KANModule::KANModule(size_t input_dim, size_t output_dim, size_t num_knots)
    : input_dim_(input_dim)
    , output_dim_(output_dim)
{
    if (input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument("KANModule requires non-zero dimensions");
    }
    
    // Create one layer per output dimension
    layers_.reserve(output_dim);
    for (size_t i = 0; i < output_dim; i++) {
        layers_.push_back(std::make_unique<KANLayer>(input_dim, num_knots));
    }
}

std::vector<double> KANModule::evaluate(const std::vector<double>& inputs) const {
    if (inputs.size() != input_dim_) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    
    std::vector<double> outputs(output_dim_);
    
    for (size_t out_idx = 0; out_idx < output_dim_; out_idx++) {
        // Each layer produces node outputs, sum them
        auto node_outputs = layers_[out_idx]->evaluate(inputs);
        double sum = 0.0;
        for (double val : node_outputs) {
            sum += val;
        }
        outputs[out_idx] = sum;
    }
    
    return outputs;
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
        
        if (config.verbose && iter % 100 == 0) {
            // Note: Would print here but avoiding IO
        }
        
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
    const double epsilon = 1e-6;
    
    // For each output dimension
    for (size_t out_idx = 0; out_idx < output_dim_; out_idx++) {
        auto& layer = layers_[out_idx];
        auto& nodes = layer->get_nodes_mutable();
        
        // For each node in layer
        for (size_t node_idx = 0; node_idx < nodes.size(); node_idx++) {
            auto& node = nodes[node_idx];
            auto coefs = node->get_coefficients();
            std::vector<double> gradient(coefs.size(), 0.0);
            
            // Accumulate gradients over dataset
            for (const auto& point : dataset) {
                auto predictions = evaluate(point.inputs);
                double error = predictions[out_idx] - point.outputs[out_idx];
                
                // Compute gradient of node output w.r.t. coefficients
                auto node_grad = node->gradient(point.inputs[node_idx], epsilon);
                
                for (size_t c = 0; c < coefs.size(); c++) {
                    gradient[c] += 2.0 * error * node_grad[c];
                }
            }
            
            // Average gradient and update
            for (size_t c = 0; c < coefs.size(); c++) {
                gradient[c] /= static_cast<double>(dataset.size());
                coefs[c] -= learning_rate * gradient[c];
            }
            
            node->set_coefficients(coefs);
        }
    }
}

} // namespace brain19
