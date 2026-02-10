#pragma once

// Lock hierarchy (always acquire in this order to prevent deadlock):
// 1. SharedLTM
// 2. SharedSTM
// 3. SharedRegistry
// 4. SharedEmbeddings

#include "../ltm/long_term_memory.hpp"
#include <mutex>
#include <shared_mutex>

namespace brain19 {

// Thread-safe wrapper around LongTermMemory.
// Uses shared_mutex: read ops get shared_lock, write ops get unique_lock.
// OPT-IN: single-threaded code can use LongTermMemory directly.
class SharedLTM {
public:
    explicit SharedLTM(LongTermMemory& ltm) : ltm_(ltm) {}

    // Non-copyable, non-movable (holds reference + mutex)
    SharedLTM(const SharedLTM&) = delete;
    SharedLTM& operator=(const SharedLTM&) = delete;

    // === WRITE operations (unique_lock) ===

    ConceptId store_concept(const std::string& label, const std::string& definition,
                            EpistemicMetadata epistemic) {
        std::unique_lock lock(mtx_);
        return ltm_.store_concept(label, definition, std::move(epistemic));
    }

    bool update_epistemic_metadata(ConceptId id, EpistemicMetadata new_metadata) {
        std::unique_lock lock(mtx_);
        return ltm_.update_epistemic_metadata(id, std::move(new_metadata));
    }

    bool invalidate_concept(ConceptId id, double invalidation_trust = 0.05) {
        std::unique_lock lock(mtx_);
        return ltm_.invalidate_concept(id, invalidation_trust);
    }

    RelationId add_relation(ConceptId source, ConceptId target, RelationType type, double weight = 1.0) {
        std::unique_lock lock(mtx_);
        return ltm_.add_relation(source, target, type, weight);
    }

    bool remove_relation(RelationId id) {
        std::unique_lock lock(mtx_);
        return ltm_.remove_relation(id);
    }

    // === READ operations (shared_lock) ===

    std::optional<ConceptInfo> retrieve_concept(ConceptId id) const {
        std::shared_lock lock(mtx_);
        return ltm_.retrieve_concept(id);
    }

    bool exists(ConceptId id) const {
        std::shared_lock lock(mtx_);
        return ltm_.exists(id);
    }

    std::vector<ConceptId> get_concepts_by_type(EpistemicType type) const {
        std::shared_lock lock(mtx_);
        return ltm_.get_concepts_by_type(type);
    }

    std::vector<ConceptId> get_concepts_by_status(EpistemicStatus status) const {
        std::shared_lock lock(mtx_);
        return ltm_.get_concepts_by_status(status);
    }

    std::vector<ConceptId> get_active_concepts() const {
        std::shared_lock lock(mtx_);
        return ltm_.get_active_concepts();
    }

    std::optional<RelationInfo> get_relation(RelationId id) const {
        std::shared_lock lock(mtx_);
        return ltm_.get_relation(id);
    }

    std::vector<RelationInfo> get_outgoing_relations(ConceptId source) const {
        std::shared_lock lock(mtx_);
        return ltm_.get_outgoing_relations(source);
    }

    std::vector<RelationInfo> get_incoming_relations(ConceptId target) const {
        std::shared_lock lock(mtx_);
        return ltm_.get_incoming_relations(target);
    }

    std::vector<RelationInfo> get_relations_between(ConceptId source, ConceptId target) const {
        std::shared_lock lock(mtx_);
        return ltm_.get_relations_between(source, target);
    }

    size_t get_relation_count(ConceptId concept_id) const {
        std::shared_lock lock(mtx_);
        return ltm_.get_relation_count(concept_id);
    }

    std::vector<ConceptId> get_all_concept_ids() const {
        std::shared_lock lock(mtx_);
        return ltm_.get_all_concept_ids();
    }

private:
    LongTermMemory& ltm_;
    mutable std::shared_mutex mtx_;
};

} // namespace brain19
