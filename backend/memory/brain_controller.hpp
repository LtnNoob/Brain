#pragma once

#include "stm.hpp"
#include <memory>
#include <chrono>

namespace brain19 {

// BrainController: Minimal orchestration layer
// DOES: Context management, flow coordination
// DOES NOT: Learn, reason, evaluate, decide importance
class BrainController {
public:
    BrainController();
    ~BrainController();
    
    BrainController(const BrainController&) = delete;
    BrainController& operator=(const BrainController&) = delete;
    BrainController(BrainController&&) = delete;
    BrainController& operator=(BrainController&&) = delete;
    
    // Initialization
    bool initialize();
    void shutdown();
    bool is_initialized() const { return initialized_; }
    
    // Context lifecycle (explicit only)
    ContextId create_context();
    void destroy_context(ContextId context_id);
    
    // Thinking lifecycle (mechanical)
    void begin_thinking(ContextId context_id);
    void end_thinking(ContextId context_id);
    
    // Explicit activation delegation (caller decides what/when)
    void activate_concept_in_context(ContextId context_id, ConceptId concept_id,
                                     double activation, ActivationClass classification);
    void activate_relation_in_context(ContextId context_id, ConceptId source,
                                      ConceptId target, RelationType type, double activation);
    
    // Explicit decay delegation (caller decides when)
    void decay_context(ContextId context_id, double time_delta_seconds);
    
    // Query delegation
    double query_concept_activation(ContextId context_id, ConceptId concept_id) const;
    std::vector<ConceptId> query_active_concepts(ContextId context_id, double threshold) const;
    
    // Access to subsystems (const only)
    const ShortTermMemory* get_stm() const { return stm_.get(); }

    // Mutable access for Cognitive Dynamics (use with caution)
    ShortTermMemory* get_stm_mutable() { return stm_.get(); }

private:
    std::unique_ptr<ShortTermMemory> stm_;
    bool initialized_ = false;
    
    // Thinking state per context
    struct ThinkingState {
        bool is_thinking = false;
        std::chrono::steady_clock::time_point start_time{};
    };
    std::unordered_map<ContextId, ThinkingState> thinking_states_;
};

} // namespace brain19
