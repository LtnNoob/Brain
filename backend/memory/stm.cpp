#include "stm.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace brain19 {

ShortTermMemory::ShortTermMemory()
    : next_context_id_(1)
    , core_decay_rate_(0.05)
    , contextual_decay_rate_(0.15)
    , relation_decay_rate_(0.25)
    , relation_inactive_threshold_(0.1)   // ε
    , relation_removal_threshold_(0.01)   // ε₂
    , concept_removal_threshold_(0.01)    // ε₃ (same default as relation)
{
}

ShortTermMemory::~ShortTermMemory() {
    contexts_.clear();
}

ContextId ShortTermMemory::create_context() {
    ContextId id = next_context_id_++;
    contexts_[id] = Context();
    return id;
}

void ShortTermMemory::destroy_context(ContextId context_id) {
    contexts_.erase(context_id);
}

void ShortTermMemory::clear_context(ContextId context_id) {
    auto it = contexts_.find(context_id);
    if (it != contexts_.end()) {
        it->second.concepts.clear();
        it->second.relations.clear();
    }
}

void ShortTermMemory::activate_concept(
    ContextId context_id,
    ConceptId concept_id,
    double activation,
    ActivationClass classification
) {
    auto it = contexts_.find(context_id);
    if (it == contexts_.end()) {
        return;
    }
    
    activation = clamp_activation(activation);
    STMEntry entry(concept_id, activation, classification);
    it->second.concepts[concept_id] = entry;
}

void ShortTermMemory::activate_relation(
    ContextId context_id,
    ConceptId source,
    ConceptId target,
    RelationType type,
    double activation
) {
    auto it = contexts_.find(context_id);
    if (it == contexts_.end()) {
        return;
    }
    
    activation = clamp_activation(activation);
    uint64_t hash = hash_relation(source, target);
    ActiveRelation relation(source, target, type, activation);
    it->second.relations[hash] = relation;
}

void ShortTermMemory::boost_concept(
    ContextId context_id,
    ConceptId concept_id,
    double delta
) {
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return;
    }
    
    auto concept_it = ctx_it->second.concepts.find(concept_id);
    if (concept_it != ctx_it->second.concepts.end()) {
        concept_it->second.activation = clamp_activation(
            concept_it->second.activation + delta
        );
        concept_it->second.last_used = std::chrono::steady_clock::now();
    }
}

void ShortTermMemory::boost_relation(
    ContextId context_id,
    ConceptId source,
    ConceptId target,
    double delta
) {
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return;
    }
    
    uint64_t hash = hash_relation(source, target);
    auto rel_it = ctx_it->second.relations.find(hash);
    if (rel_it != ctx_it->second.relations.end()) {
        rel_it->second.activation = clamp_activation(
            rel_it->second.activation + delta
        );
        rel_it->second.last_used = std::chrono::steady_clock::now();
    }
}

double ShortTermMemory::get_concept_activation(
    ContextId context_id,
    ConceptId concept_id
) const {
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return 0.0;
    }
    
    auto concept_it = ctx_it->second.concepts.find(concept_id);
    if (concept_it != ctx_it->second.concepts.end()) {
        return concept_it->second.activation;
    }
    return 0.0;
}

double ShortTermMemory::get_relation_activation(
    ContextId context_id,
    ConceptId source,
    ConceptId target
) const {
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return 0.0;
    }
    
    uint64_t hash = hash_relation(source, target);
    auto rel_it = ctx_it->second.relations.find(hash);
    if (rel_it != ctx_it->second.relations.end()) {
        return rel_it->second.activation;
    }
    return 0.0;
}

ActivationLevel ShortTermMemory::get_concept_level(
    ContextId context_id,
    ConceptId concept_id
) const {
    double activation = get_concept_activation(context_id, concept_id);
    return classify_level(activation);
}

std::vector<ConceptId> ShortTermMemory::get_active_concepts(
    ContextId context_id,
    double threshold
) const {
    std::vector<ConceptId> result;
    
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return result;
    }
    
    for (const auto& pair : ctx_it->second.concepts) {
        if (pair.second.activation >= threshold) {
            result.push_back(pair.first);
        }
    }
    
    return result;
}

std::vector<ActiveRelation> ShortTermMemory::get_active_relations(
    ContextId context_id,
    double threshold
) const {
    std::vector<ActiveRelation> result;
    
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return result;
    }
    
    for (const auto& pair : ctx_it->second.relations) {
        if (pair.second.activation >= threshold) {
            result.push_back(pair.second);
        }
    }
    
    return result;
}

void ShortTermMemory::decay_all(ContextId context_id, double time_delta_seconds) {
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return;
    }
    
    // Decay concepts
    for (auto& pair : ctx_it->second.concepts) {
        apply_decay(pair.second, time_delta_seconds);
    }
    
    // Remove concepts that decayed below removal threshold
    for (auto it = ctx_it->second.concepts.begin(); it != ctx_it->second.concepts.end(); ) {
        if (it->second.activation < concept_removal_threshold_) {
            it = ctx_it->second.concepts.erase(it);
        } else {
            ++it;
        }
    }
    
    // Two-phase decay for relations
    std::vector<uint64_t> to_remove;
    for (auto& pair : ctx_it->second.relations) {
        apply_relation_decay(pair.second, time_delta_seconds);
        
        // Phase 2: Remove only if below removal threshold (prevents flapping)
        if (pair.second.activation < relation_removal_threshold_) {
            to_remove.push_back(pair.first);
        }
    }
    
    for (uint64_t hash : to_remove) {
        ctx_it->second.relations.erase(hash);
    }
}

void ShortTermMemory::set_core_decay_rate(double rate) {
    core_decay_rate_ = std::max(0.0, rate);
}

void ShortTermMemory::set_contextual_decay_rate(double rate) {
    contextual_decay_rate_ = std::max(0.0, rate);
}

void ShortTermMemory::set_relation_decay_rate(double rate) {
    relation_decay_rate_ = std::max(0.0, rate);
}

void ShortTermMemory::set_relation_inactive_threshold(double threshold) {
    relation_inactive_threshold_ = clamp_activation(threshold);
}

void ShortTermMemory::set_relation_removal_threshold(double threshold) {
    relation_removal_threshold_ = clamp_activation(threshold);
}

void ShortTermMemory::set_concept_removal_threshold(double threshold) {
    concept_removal_threshold_ = clamp_activation(threshold);
}

size_t ShortTermMemory::debug_active_concept_count(ContextId context_id) const {
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return 0;
    }
    return ctx_it->second.concepts.size();
}

size_t ShortTermMemory::debug_active_relation_count(ContextId context_id) const {
    auto ctx_it = contexts_.find(context_id);
    if (ctx_it == contexts_.end()) {
        return 0;
    }
    return ctx_it->second.relations.size();
}

STMSnapshotData ShortTermMemory::export_state() const {
    STMSnapshotData data;
    data.timestamp = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    for (const auto& [ctx_id, ctx] : contexts_) {
        SnapshotContext sc;
        sc.context_id = ctx_id;
        for (const auto& [cid, entry] : ctx.concepts) {
            sc.concepts.push_back({entry.concept_id, entry.activation, entry.classification});
        }
        for (const auto& [hash, rel] : ctx.relations) {
            sc.relations.push_back({rel.source, rel.target, rel.type, rel.activation});
        }
        data.contexts.push_back(std::move(sc));
    }
    return data;
}

void ShortTermMemory::import_state(const STMSnapshotData& data) {
    contexts_.clear();
    for (const auto& sc : data.contexts) {
        if (sc.context_id >= next_context_id_) {
            next_context_id_ = sc.context_id + 1;
        }
        Context& ctx = contexts_[sc.context_id];
        for (const auto& c : sc.concepts) {
            ctx.concepts[c.concept_id] = STMEntry(c.concept_id, c.activation, c.classification);
        }
        for (const auto& r : sc.relations) {
            uint64_t h = hash_relation(r.source, r.target);
            ctx.relations[h] = ActiveRelation(r.source, r.target, r.type, r.activation);
        }
    }
}

double ShortTermMemory::clamp_activation(double value) const {
    return std::max(0.0, std::min(1.0, value));
}

ActivationLevel ShortTermMemory::classify_level(double activation) const {
    if (activation < 0.3) {
        return ActivationLevel::LOW;
    } else if (activation < 0.7) {
        return ActivationLevel::MEDIUM;
    } else {
        return ActivationLevel::HIGH;
    }
}

uint64_t ShortTermMemory::hash_relation(ConceptId source, ConceptId target) const {
    // hash_combine pattern: handles full 64-bit ConceptIds without truncation
    uint64_t h = std::hash<uint64_t>{}(source);
    h ^= std::hash<uint64_t>{}(target) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

void ShortTermMemory::apply_decay(STMEntry& entry, double time_delta) const {
    double rate = (entry.classification == ActivationClass::CORE_KNOWLEDGE)
        ? core_decay_rate_
        : contextual_decay_rate_;
    
    double decay_factor = std::exp(-rate * time_delta);
    entry.activation = clamp_activation(entry.activation * decay_factor);
}

void ShortTermMemory::apply_relation_decay(ActiveRelation& relation, double time_delta) const {
    double decay_factor = std::exp(-relation_decay_rate_ * time_delta);
    relation.activation = clamp_activation(relation.activation * decay_factor);
}

} // namespace brain19
