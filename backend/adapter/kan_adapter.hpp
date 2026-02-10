#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace brain19 {

// Forward declarations
class KANModule;
struct FunctionHypothesis;
struct DataPoint;
struct TrainingConfig;

// KANAdapter: Clean interface between BrainController and KAN
// Provides explicit delegation, NO decision logic
class KANAdapter {
public:
    KANAdapter();
    ~KANAdapter();
    
    // Create new KAN module
    // Returns module ID for later reference
    uint64_t create_kan_module(size_t input_dim, size_t output_dim, size_t num_knots = 10);
    
    // Train existing KAN module
    // Returns FunctionHypothesis (pure data wrapper)
    std::unique_ptr<FunctionHypothesis> train_kan_module(
        uint64_t module_id,
        const std::vector<DataPoint>& dataset,
        const TrainingConfig& config
    );
    
    // Evaluate KAN module
    std::vector<double> evaluate_kan_module(
        uint64_t module_id,
        const std::vector<double>& inputs
    ) const;
    
    // Destroy module (explicit cleanup)
    void destroy_kan_module(uint64_t module_id);
    
    // Query if module exists
    bool has_module(uint64_t module_id) const;
    
private:
    struct KANModuleEntry {
        std::shared_ptr<KANModule> module;
        size_t input_dim;
        size_t output_dim;
    };
    
    std::unordered_map<uint64_t, KANModuleEntry> modules_;
    uint64_t next_module_id_;
};

} // namespace brain19
