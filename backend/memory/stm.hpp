#pragma once

#include "activation_level.hpp"
#include "stm_entry.hpp"
#include "active_relation.hpp"
#include <unordered_map>
#include <vector>

namespace brain19 {

using ContextId = uint64_t;

// Short-Term Memory: Purely mechanical activation layer
// INVARIANT: STM never stores knowledge, only activation states
// INVARIANT: STM never evaluates correctness, trust, or importance
class ShortTermMemory {
public:
    ShortTermMemory();
    ~ShortTermMemory();
    
    // Context management
    ContextId create_context();
    void destroy_context(ContextId context_id);
    void clear_context(ContextId context_id);
    
    // Explicit activation (caller decides what/when)
    void activate_concept(ContextId context_id, ConceptId concept_id, 
                         double activation, ActivationClass classification);
    void activate_relation(ContextId context_id, ConceptId source, ConceptId target,
                          RelationType type, double activation);
    
    // Boost existing activation
    void boost_concept(ContextId context_id, ConceptId concept_id, double delta);
    void boost_relation(ContextId context_id, ConceptId source, ConceptId target, double delta);
    
    // Query activation state
    double get_concept_activation(ContextId context_id, ConceptId concept_id) const;
    double get_relation_activation(ContextId context_id, ConceptId source, ConceptId target) const;
    ActivationLevel get_concept_level(ContextId context_id, ConceptId concept_id) const;
    
    // Get activated items above threshold
    std::vector<ConceptId> get_active_concepts(ContextId context_id, double threshold = 0.0) const;
    std::vector<ActiveRelation> get_active_relations(ContextId context_id, double threshold = 0.0) const;
    
    // Mechanical decay (explicit call only)
    void decay_all(ContextId context_id, double time_delta_seconds);
    
    // Configuration
    void set_core_decay_rate(double rate);
    void set_contextual_decay_rate(double rate);
    void set_relation_decay_rate(double rate);
    void set_relation_inactive_threshold(double threshold);
    void set_relation_removal_threshold(double threshold);
    
    // Debug introspection (NOT for operational logic)
    size_t debug_active_concept_count(ContextId context_id) const;
    size_t debug_active_relation_count(ContextId context_id) const;
    
private:
    struct Context {
        std::unordered_map<ConceptId, STMEntry> concepts;
        std::unordered_map<uint64_t, ActiveRelation> relations;
    };
    
    std::unordered_map<ContextId, Context> contexts_;
    ContextId next_context_id_;
    
    double core_decay_rate_;
    double contextual_decay_rate_;
    double relation_decay_rate_;
    double relation_inactive_threshold_;  // ε
    double relation_removal_threshold_;   // ε₂
    
    double clamp_activation(double value) const;
    ActivationLevel classify_level(double activation) const;
    uint64_t hash_relation(ConceptId source, ConceptId target) const;
    void apply_decay(STMEntry& entry, double time_delta) const;
    void apply_relation_decay(ActiveRelation& relation, double time_delta) const;
};

} // namespace brain19
