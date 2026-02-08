#include "long_term_memory.hpp"
#include <algorithm>
#include <vector>

namespace brain19 {

LongTermMemory::LongTermMemory()
    : next_concept_id_(1)
{
}

LongTermMemory::~LongTermMemory() {
}

ConceptId LongTermMemory::store_concept(
    const std::string& label,
    const std::string& definition,
    EpistemicMetadata epistemic
) {
    // Generate unique ID
    ConceptId id = next_concept_id_++;
    
    // Create ConceptInfo with required epistemic metadata
    // This constructor call enforces epistemic explicitness
    ConceptInfo info(id, label, definition, epistemic);
    
    // Store using emplace (avoids default construction)
    concepts_.emplace(id, std::move(info));
    
    return id;
}

std::optional<ConceptInfo> LongTermMemory::retrieve_concept(ConceptId id) const {
    auto it = concepts_.find(id);
    if (it == concepts_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool LongTermMemory::exists(ConceptId id) const {
    return concepts_.find(id) != concepts_.end();
}

bool LongTermMemory::update_epistemic_metadata(
    ConceptId id,
    EpistemicMetadata new_metadata
) {
    auto it = concepts_.find(id);
    if (it == concepts_.end()) {
        return false;
    }
    
    // Extract current data
    std::string label = it->second.label;
    std::string definition = it->second.definition;
    
    // Erase old entry
    concepts_.erase(it);
    
    // Create new ConceptInfo with updated metadata
    ConceptInfo updated(id, label, definition, new_metadata);
    
    // Insert updated entry
    concepts_.emplace(id, std::move(updated));
    
    return true;
}

bool LongTermMemory::invalidate_concept(ConceptId id, double invalidation_trust) {
    auto it = concepts_.find(id);
    if (it == concepts_.end()) {
        return false;
    }
    
    // CRITICAL: Knowledge is NOT deleted, only invalidated
    // Original type is preserved
    // Status set to INVALIDATED
    // Trust set very low
    
    // Validate invalidation trust
    if (invalidation_trust < 0.0 || invalidation_trust > 1.0) {
        invalidation_trust = 0.05;  // Default
    }
    
    // Create invalidated metadata
    EpistemicMetadata invalidated_meta(
        it->second.epistemic.type,           // Preserve original type
        EpistemicStatus::INVALIDATED,        // Mark as invalidated
        invalidation_trust                   // Very low trust
    );
    
    return update_epistemic_metadata(id, invalidated_meta);
}

std::vector<ConceptId> LongTermMemory::get_concepts_by_type(EpistemicType type) const {
    std::vector<ConceptId> result;
    
    for (const auto& pair : concepts_) {
        if (pair.second.epistemic.type == type) {
            result.push_back(pair.first);
        }
    }
    
    return result;
}

std::vector<ConceptId> LongTermMemory::get_concepts_by_status(EpistemicStatus status) const {
    std::vector<ConceptId> result;
    
    for (const auto& pair : concepts_) {
        if (pair.second.epistemic.status == status) {
            result.push_back(pair.first);
        }
    }
    
    return result;
}

std::vector<ConceptId> LongTermMemory::get_active_concepts() const {
    return get_concepts_by_status(EpistemicStatus::ACTIVE);
}

// =============================================================================
// RELATION MANAGEMENT IMPLEMENTATION
// =============================================================================

RelationId LongTermMemory::add_relation(
    ConceptId source,
    ConceptId target,
    RelationType type,
    double weight
) {
    // Validate concepts exist
    if (!exists(source) || !exists(target)) {
        return 0;  // Invalid relation
    }

    RelationId id = next_relation_id_++;

    // Create relation (RelationInfo constructor clamps weight)
    relations_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(id),
        std::forward_as_tuple(id, source, target, type, weight)
    );

    // Update indices
    outgoing_relations_[source].push_back(id);
    incoming_relations_[target].push_back(id);

    return id;
}

std::optional<RelationInfo> LongTermMemory::get_relation(RelationId id) const {
    auto it = relations_.find(id);
    if (it == relations_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<RelationInfo> LongTermMemory::get_outgoing_relations(ConceptId source) const {
    std::vector<RelationInfo> result;

    auto it = outgoing_relations_.find(source);
    if (it != outgoing_relations_.end()) {
        for (RelationId rid : it->second) {
            auto rel_it = relations_.find(rid);
            if (rel_it != relations_.end()) {
                result.push_back(rel_it->second);
            }
        }
    }

    return result;
}

std::vector<RelationInfo> LongTermMemory::get_incoming_relations(ConceptId target) const {
    std::vector<RelationInfo> result;

    auto it = incoming_relations_.find(target);
    if (it != incoming_relations_.end()) {
        for (RelationId rid : it->second) {
            auto rel_it = relations_.find(rid);
            if (rel_it != relations_.end()) {
                result.push_back(rel_it->second);
            }
        }
    }

    return result;
}

std::vector<RelationInfo> LongTermMemory::get_relations_between(ConceptId source, ConceptId target) const {
    std::vector<RelationInfo> result;

    auto outgoing = get_outgoing_relations(source);
    for (const auto& rel : outgoing) {
        if (rel.target == target) {
            result.push_back(rel);
        }
    }

    return result;
}

bool LongTermMemory::remove_relation(RelationId id) {
    auto it = relations_.find(id);
    if (it == relations_.end()) {
        return false;
    }

    ConceptId source = it->second.source;
    ConceptId target = it->second.target;

    // Remove from indices
    auto& out_rels = outgoing_relations_[source];
    out_rels.erase(std::remove(out_rels.begin(), out_rels.end(), id), out_rels.end());

    auto& in_rels = incoming_relations_[target];
    in_rels.erase(std::remove(in_rels.begin(), in_rels.end(), id), in_rels.end());

    // Remove relation itself
    relations_.erase(it);

    return true;
}

size_t LongTermMemory::get_relation_count(ConceptId concept_id) const {
    size_t count = 0;

    auto out_it = outgoing_relations_.find(concept_id);
    if (out_it != outgoing_relations_.end()) {
        count += out_it->second.size();
    }

    auto in_it = incoming_relations_.find(concept_id);
    if (in_it != incoming_relations_.end()) {
        count += in_it->second.size();
    }

    return count;
}

std::vector<ConceptId> LongTermMemory::get_all_concept_ids() const {
    std::vector<ConceptId> result;
    result.reserve(concepts_.size());

    for (const auto& pair : concepts_) {
        result.push_back(pair.first);
    }

    return result;
}

} // namespace brain19
