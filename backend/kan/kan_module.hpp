#pragma once

#include "kan_layer.hpp"
#include <vector>
#include <chrono>

namespace brain19 {

// Training data point
struct DataPoint {
    std::vector<double> inputs;
    std::vector<double> outputs;
    
    DataPoint() = default;
    DataPoint(const std::vector<double>& in, const std::vector<double>& out)
        : inputs(in), outputs(out) {}
};

// Training configuration
struct KanTrainingConfig {
    size_t max_iterations = 1000;
    double learning_rate = 0.01;
    double convergence_threshold = 1e-6;
    bool verbose = false;
};

// Training result
struct KanTrainingResult {
    size_t iterations_run;
    double final_loss;
    bool converged;
    std::chrono::milliseconds duration;
};

// KANModule: Complete functional approximator using multi-layer KAN
// Represents f: R^n -> R^m via a chain of KANLayers
//
// Layer topology example: [3, 5, 2] means:
//   Layer 0: 3 inputs → 5 outputs (3×5 = 15 nodes)
//   Layer 1: 5 inputs → 2 outputs (5×2 = 10 nodes)
// Overall: R^3 → R^2
class KANModule {
public:
    // Multi-layer constructor with explicit topology
    // layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
    // Must have at least 2 elements (input + output)
    explicit KANModule(const std::vector<size_t>& layer_dims, size_t num_knots = 10);
    
    // Legacy single-layer constructor (backward compatible)
    // Creates topology [input_dim, output_dim]
    KANModule(size_t input_dim, size_t output_dim, size_t num_knots = 10);
    
    // Evaluate function (forward pass through all layers)
    std::vector<double> evaluate(const std::vector<double>& inputs) const;
    
    // Train on dataset
    KanTrainingResult train(const std::vector<DataPoint>& dataset, 
                        const KanTrainingConfig& config = KanTrainingConfig());
    
    // Validation
    double compute_mse(const std::vector<DataPoint>& dataset) const;
    
    // Deep copy for warm-start training
    std::shared_ptr<KANModule> clone() const;

    // Dimensions
    size_t input_dim() const { return input_dim_; }
    size_t output_dim() const { return output_dim_; }
    size_t num_layers() const { return layers_.size(); }
    const std::vector<size_t>& topology() const { return layer_dims_; }
    
    // Access layers (for inspection/advanced use)
    const KANLayer& layer(size_t idx) const { return *layers_[idx]; }
    KANLayer& layer_mutable(size_t idx) { return *layers_[idx]; }
    
private:
    size_t input_dim_;
    size_t output_dim_;
    std::vector<size_t> layer_dims_;  // Full topology
    
    // layers_[k] maps from layer_dims_[k] → layer_dims_[k+1]
    std::vector<std::unique_ptr<KANLayer>> layers_;
    
    // Training helpers
    double compute_loss(const std::vector<DataPoint>& dataset) const;
    void gradient_descent_step(const std::vector<DataPoint>& dataset, double learning_rate);
    
    // Forward pass returning all intermediate activations (for training)
    std::vector<std::vector<double>> forward_all(const std::vector<double>& inputs) const;
};

} // namespace brain19
