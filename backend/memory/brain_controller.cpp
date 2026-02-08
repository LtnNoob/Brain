#include "brain_controller.hpp"

namespace brain19 {

BrainController::BrainController()
    : initialized_(false)
{
}

BrainController::~BrainController() {
    if (initialized_) {
        shutdown();
    }
}

bool BrainController::initialize() {
    if (initialized_) {
        return true;
    }
    
    stm_ = std::make_unique<ShortTermMemory>();
    
    initialized_ = true;
    return true;
}

void BrainController::shutdown() {
    if (!initialized_) {
        return;
    }
    
    thinking_states_.clear();
    stm_.reset();
    
    initialized_ = false;
}

ContextId BrainController::create_context() {
    if (!initialized_) {
        return 0;
    }
    
    ContextId ctx = stm_->create_context();
    
    ThinkingState state;
    state.is_thinking = false;
    thinking_states_[ctx] = state;
    
    return ctx;
}

void BrainController::destroy_context(ContextId context_id) {
    if (!initialized_) {
        return;
    }
    
    thinking_states_.erase(context_id);
    stm_->destroy_context(context_id);
}

void BrainController::begin_thinking(ContextId context_id) {
    if (!initialized_) {
        return;
    }
    
    auto it = thinking_states_.find(context_id);
    if (it != thinking_states_.end()) {
        it->second.is_thinking = true;
        it->second.start_time = std::chrono::steady_clock::now();
    }
}

void BrainController::end_thinking(ContextId context_id) {
    if (!initialized_) {
        return;
    }
    
    auto it = thinking_states_.find(context_id);
    if (it != thinking_states_.end()) {
        it->second.is_thinking = false;
    }
}

void BrainController::activate_concept_in_context(
    ContextId context_id,
    ConceptId concept_id,
    double activation,
    ActivationClass classification
) {
    if (!initialized_) {
        return;
    }
    
    stm_->activate_concept(context_id, concept_id, activation, classification);
}

void BrainController::activate_relation_in_context(
    ContextId context_id,
    ConceptId source,
    ConceptId target,
    RelationType type,
    double activation
) {
    if (!initialized_) {
        return;
    }
    
    stm_->activate_relation(context_id, source, target, type, activation);
}

void BrainController::decay_context(ContextId context_id, double time_delta_seconds) {
    if (!initialized_) {
        return;
    }
    
    stm_->decay_all(context_id, time_delta_seconds);
}

double BrainController::query_concept_activation(
    ContextId context_id,
    ConceptId concept_id
) const {
    if (!initialized_) {
        return 0.0;
    }
    
    return stm_->get_concept_activation(context_id, concept_id);
}

std::vector<ConceptId> BrainController::query_active_concepts(
    ContextId context_id,
    double threshold
) const {
    if (!initialized_) {
        return {};
    }
    
    return stm_->get_active_concepts(context_id, threshold);
}

} // namespace brain19
