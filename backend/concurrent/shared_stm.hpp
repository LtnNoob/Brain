#pragma once

// Lock hierarchy (always acquire in this order to prevent deadlock):
// 1. SharedLTM
// 2. SharedSTM
// 3. SharedRegistry
// 4. SharedEmbeddings

#include "../memory/stm.hpp"
#include <shared_mutex>
#include <mutex>
#include <unordered_map>

namespace brain19 {

// Thread-safe wrapper around ShortTermMemory.
// Per-context mutexes allow parallel access to different contexts.
// Global mutex for cross-context and config operations.
// OPT-IN: single-threaded code can use ShortTermMemory directly.
class SharedSTM {
public:
    explicit SharedSTM(ShortTermMemory& stm) : stm_(stm) {}

    SharedSTM(const SharedSTM&) = delete;
    SharedSTM& operator=(const SharedSTM&) = delete;

    // === Context management (global lock) ===

    ContextId create_context() {
        std::unique_lock lock(global_mtx_);
        auto cid = stm_.create_context();
        // Pre-create per-context mutex
        context_mutexes_[cid];
        return cid;
    }

    void destroy_context(ContextId context_id) {
        std::unique_lock lock(global_mtx_);
        stm_.destroy_context(context_id);
        context_mutexes_.erase(context_id);
    }

    void clear_context(ContextId context_id) {
        auto& cmtx = get_context_mutex(context_id);
        std::unique_lock lock(cmtx);
        stm_.clear_context(context_id);
    }

    // === Per-context WRITE operations ===

    void activate_concept(ContextId context_id, ConceptId concept_id,
                         double activation, ActivationClass classification) {
        auto& cmtx = get_context_mutex(context_id);
        std::unique_lock lock(cmtx);
        stm_.activate_concept(context_id, concept_id, activation, classification);
    }

    void activate_relation(ContextId context_id, ConceptId source, ConceptId target,
                          RelationType type, double activation) {
        auto& cmtx = get_context_mutex(context_id);
        std::unique_lock lock(cmtx);
        stm_.activate_relation(context_id, source, target, type, activation);
    }

    void boost_concept(ContextId context_id, ConceptId concept_id, double delta) {
        auto& cmtx = get_context_mutex(context_id);
        std::unique_lock lock(cmtx);
        stm_.boost_concept(context_id, concept_id, delta);
    }

    void boost_relation(ContextId context_id, ConceptId source, ConceptId target, double delta) {
        auto& cmtx = get_context_mutex(context_id);
        std::unique_lock lock(cmtx);
        stm_.boost_relation(context_id, source, target, delta);
    }

    // === Per-context READ operations ===

    double get_concept_activation(ContextId context_id, ConceptId concept_id) const {
        auto& cmtx = get_context_mutex(context_id);
        std::shared_lock lock(cmtx);
        return stm_.get_concept_activation(context_id, concept_id);
    }

    double get_relation_activation(ContextId context_id, ConceptId source, ConceptId target) const {
        auto& cmtx = get_context_mutex(context_id);
        std::shared_lock lock(cmtx);
        return stm_.get_relation_activation(context_id, source, target);
    }

    ActivationLevel get_concept_level(ContextId context_id, ConceptId concept_id) const {
        auto& cmtx = get_context_mutex(context_id);
        std::shared_lock lock(cmtx);
        return stm_.get_concept_level(context_id, concept_id);
    }

    std::vector<ConceptId> get_active_concepts(ContextId context_id, double threshold = 0.0) const {
        auto& cmtx = get_context_mutex(context_id);
        std::shared_lock lock(cmtx);
        return stm_.get_active_concepts(context_id, threshold);
    }

    std::vector<ActiveRelation> get_active_relations(ContextId context_id, double threshold = 0.0) const {
        auto& cmtx = get_context_mutex(context_id);
        std::shared_lock lock(cmtx);
        return stm_.get_active_relations(context_id, threshold);
    }

    // === Cross-context operations (global lock) ===

    void decay_all(ContextId context_id, double time_delta_seconds) {
        auto& cmtx = get_context_mutex(context_id);
        std::unique_lock lock(cmtx);
        stm_.decay_all(context_id, time_delta_seconds);
    }

    // === Configuration (global lock — not thread-safe to change mid-flight) ===

    void set_core_decay_rate(double rate) {
        std::unique_lock lock(global_mtx_);
        stm_.set_core_decay_rate(rate);
    }

    void set_contextual_decay_rate(double rate) {
        std::unique_lock lock(global_mtx_);
        stm_.set_contextual_decay_rate(rate);
    }

    void set_relation_decay_rate(double rate) {
        std::unique_lock lock(global_mtx_);
        stm_.set_relation_decay_rate(rate);
    }

    void set_relation_inactive_threshold(double threshold) {
        std::unique_lock lock(global_mtx_);
        stm_.set_relation_inactive_threshold(threshold);
    }

    void set_relation_removal_threshold(double threshold) {
        std::unique_lock lock(global_mtx_);
        stm_.set_relation_removal_threshold(threshold);
    }

    void set_concept_removal_threshold(double threshold) {
        std::unique_lock lock(global_mtx_);
        stm_.set_concept_removal_threshold(threshold);
    }

    // === Debug (per-context read lock) ===

    size_t debug_active_concept_count(ContextId context_id) const {
        auto& cmtx = get_context_mutex(context_id);
        std::shared_lock lock(cmtx);
        return stm_.debug_active_concept_count(context_id);
    }

    size_t debug_active_relation_count(ContextId context_id) const {
        auto& cmtx = get_context_mutex(context_id);
        std::shared_lock lock(cmtx);
        return stm_.debug_active_relation_count(context_id);
    }

private:
    ShortTermMemory& stm_;
    mutable std::shared_mutex global_mtx_;
    mutable std::unordered_map<ContextId, std::shared_mutex> context_mutexes_;

    std::shared_mutex& get_context_mutex(ContextId cid) const {
        // Context mutex must already exist (created in create_context)
        return context_mutexes_.at(cid);
    }
};

} // namespace brain19
