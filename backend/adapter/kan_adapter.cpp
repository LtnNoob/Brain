#include "kan_adapter.hpp"
#include "../kan/kan_module.hpp"
#include "../kan/function_hypothesis.hpp"
#include <stdexcept>

namespace brain19 {

KANAdapter::KANAdapter()
    : next_module_id_(1)
{
}

KANAdapter::~KANAdapter() {
    modules_.clear();
}

uint64_t KANAdapter::create_kan_module(size_t input_dim, size_t output_dim, size_t num_knots) {
    uint64_t id = next_module_id_++;
    
    KANModuleEntry entry;
    entry.module = std::make_shared<KANModule>(input_dim, output_dim, num_knots);
    entry.input_dim = input_dim;
    entry.output_dim = output_dim;
    
    modules_[id] = std::move(entry);
    
    return id;
}

uint64_t KANAdapter::create_kan_module_multilayer(const std::vector<size_t>& layer_dims, size_t num_knots) {
    uint64_t id = next_module_id_++;
    
    KANModuleEntry entry;
    entry.module = std::make_shared<KANModule>(layer_dims, num_knots);
    entry.input_dim = layer_dims.front();
    entry.output_dim = layer_dims.back();
    
    modules_[id] = std::move(entry);
    
    return id;
}

std::unique_ptr<FunctionHypothesis> KANAdapter::train_kan_module(
    uint64_t module_id,
    const std::vector<DataPoint>& dataset,
    const KanTrainingConfig& config
) {
    auto it = modules_.find(module_id);
    if (it == modules_.end()) {
        return nullptr;
    }
    
    auto& entry = it->second;
    auto result = entry.module->train(dataset, config);
    
    // Create FunctionHypothesis wrapper
    auto hypothesis = std::make_unique<FunctionHypothesis>(
        entry.input_dim,
        entry.output_dim,
        entry.module,  // Non-owning share
        result.iterations_run,
        result.final_loss
    );
    
    return hypothesis;
}

std::vector<double> KANAdapter::evaluate_kan_module(
    uint64_t module_id,
    const std::vector<double>& inputs
) const {
    auto it = modules_.find(module_id);
    if (it == modules_.end()) {
        return {};
    }
    
    return it->second.module->evaluate(inputs);
}

void KANAdapter::destroy_kan_module(uint64_t module_id) {
    modules_.erase(module_id);
}

bool KANAdapter::has_module(uint64_t module_id) const {
    return modules_.find(module_id) != modules_.end();
}

std::vector<size_t> KANAdapter::get_topology(uint64_t module_id) const {
    auto it = modules_.find(module_id);
    if (it == modules_.end()) {
        return {};
    }
    return it->second.module->topology();
}

} // namespace brain19
