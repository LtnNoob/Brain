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

// KANModule: Complete functional approximator
// Represents f: R^n -> R^m
class KANModule {
public:
    // Constructor: dimensions define the function signature
    KANModule(size_t input_dim, size_t output_dim, size_t num_knots = 10);
    
    // Evaluate function
    std::vector<double> evaluate(const std::vector<double>& inputs) const;
    
    // Train on dataset
    KanTrainingResult train(const std::vector<DataPoint>& dataset, 
                        const KanTrainingConfig& config = KanTrainingConfig());
    
    // Validation
    double compute_mse(const std::vector<DataPoint>& dataset) const;
    
    // Dimensions
    size_t input_dim() const { return input_dim_; }
    size_t output_dim() const { return output_dim_; }
    
private:
    size_t input_dim_;
    size_t output_dim_;
    
    // One layer per output dimension
    std::vector<std::unique_ptr<KANLayer>> layers_;
    
    // Training helpers
    double compute_loss(const std::vector<DataPoint>& dataset) const;
    void gradient_descent_step(const std::vector<DataPoint>& dataset, double learning_rate);
};

} // namespace brain19
