#pragma once

#include "../epistemic/epistemic_metadata.hpp"
#include "relation.hpp"
#include <string>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>
#include <memory>
#include "../common/types.hpp"

namespace brain19 {


// ConceptInfo: Knowledge item with MANDATORY epistemic metadata
//
// CRITICAL INVARIANT:
// - Every ConceptInfo MUST have valid EpistemicMetadata
// - Construction without epistemic metadata is IMPOSSIBLE
// - No default construction allowed
struct ConceptInfo {
    ConceptId id;
    std::string label;
    std::string definition;
    EpistemicMetadata epistemic;  // REQUIRED, no default
    
    // DELETED: No default constructor
    // This enforces epistemic explicitness at compile time
    ConceptInfo() = delete;
    
    // REQUIRED: Constructor with epistemic metadata
    ConceptInfo(
        ConceptId concept_id,
        const std::string& concept_label,
        const std::string& concept_definition,
        EpistemicMetadata epistemic_metadata  // MUST be provided
    ) : id(concept_id)
      , label(concept_label)
      , definition(concept_definition)
      , epistemic(epistemic_metadata)
    {
        // Epistemic metadata is validated in EpistemicMetadata constructor
    }
    
    // Copy/move allowed for storage
    ConceptInfo(const ConceptInfo&) = default;
    ConceptInfo(ConceptInfo&&) = default;
    
    // Assignment operators - need custom implementation
    // because EpistemicMetadata has deleted assignment
    ConceptInfo& operator=(const ConceptInfo& other) {
        if (this != &other) {
            // Recreate with new metadata
            id = other.id;
            label = other.label;
            definition = other.definition;
            // Can't assign epistemic directly (deleted), so use placement new
            epistemic.~EpistemicMetadata();
            new (&epistemic) EpistemicMetadata(other.epistemic);
        }
        return *this;
    }
    
    ConceptInfo& operator=(ConceptInfo&& other) {
        if (this != &other) {
            id = other.id;
            label = std::move(other.label);
            definition = std::move(other.definition);
            // Can't assign epistemic directly (deleted), so use placement new
            epistemic.~EpistemicMetadata();
            new (&epistemic) EpistemicMetadata(other.epistemic);
        }
        return *this;
    }
};

// LongTermMemory: Persistent knowledge storage
//
// EPISTEMIC ENFORCEMENT:
// - ALL storage methods REQUIRE explicit EpistemicMetadata
// - NO implicit defaults
// - NO silent fallbacks
// - Epistemic decisions MUST be explicit
class LongTermMemory {
public:
    LongTermMemory();
    ~LongTermMemory();
    
    // Store concept with REQUIRED epistemic metadata
    // 
    // CRITICAL: This method signature enforces epistemic explicitness
    // - EpistemicMetadata parameter has NO default
    // - Calling without it is a compile error
    // - This makes epistemic decisions visible and explicit
    ConceptId store_concept(
        const std::string& label,
        const std::string& definition,
        EpistemicMetadata epistemic  // NO DEFAULT - must be explicit
    );
    
    // Retrieve concept (returns full ConceptInfo with epistemic metadata)
    std::optional<ConceptInfo> retrieve_concept(ConceptId id) const;
    
    // Check existence
    bool exists(ConceptId id) const;
    
    // Update epistemic metadata (e.g., invalidation)
    //
    // IMPORTANT: This is the ONLY way to change epistemic status
    // - Knowledge is NEVER deleted
    // - INVALIDATED knowledge remains in storage
    // - This preserves epistemic history
    bool update_epistemic_metadata(ConceptId id, EpistemicMetadata new_metadata);
    
    // Invalidate knowledge (special case of update)
    //
    // CRITICAL: Invalidation does NOT delete
    // - Sets status to INVALIDATED
    // - Sets trust very low (< 0.2)
    // - Preserves original type
    // - Knowledge remains queryable
    bool invalidate_concept(ConceptId id, double invalidation_trust = 0.05);
    
    // Query by epistemic type
    std::vector<ConceptId> get_concepts_by_type(EpistemicType type) const;
    
    // Query by epistemic status
    std::vector<ConceptId> get_concepts_by_status(EpistemicStatus status) const;
    
    // Get all active (non-invalidated) concepts
    std::vector<ConceptId> get_active_concepts() const;

    // =========================================================================
    // RELATION MANAGEMENT (ADDITIVE EXTENSION)
    // =========================================================================
    //
    // Relations are DIRECTED edges between concepts
    // - Relations are NOT epistemic entities (no trust/type)
    // - Relations model semantic connectivity
    // - Weight affects spreading activation strength
    //

    // Add relation between concepts
    RelationId add_relation(
        ConceptId source,
        ConceptId target,
        RelationType type,
        double weight = 1.0
    );

    // Get relation by ID
    std::optional<RelationInfo> get_relation(RelationId id) const;

    // Get all outgoing relations from a concept
    std::vector<RelationInfo> get_outgoing_relations(ConceptId source) const;

    // Get all incoming relations to a concept
    std::vector<RelationInfo> get_incoming_relations(ConceptId target) const;

    // Get relations between two specific concepts
    std::vector<RelationInfo> get_relations_between(ConceptId source, ConceptId target) const;

    // Remove relation
    bool remove_relation(RelationId id);

    // Get total relation count for a concept (in + out)
    size_t get_relation_count(ConceptId concept_id) const;

    // Get all concept IDs
    std::vector<ConceptId> get_all_concept_ids() const;

private:
    std::unordered_map<ConceptId, ConceptInfo> concepts_;
    ConceptId next_concept_id_;

    // Relation storage
    std::unordered_map<RelationId, RelationInfo> relations_;
    std::unordered_map<ConceptId, std::vector<RelationId>> outgoing_relations_;
    std::unordered_map<ConceptId, std::vector<RelationId>> incoming_relations_;
    RelationId next_relation_id_ = 1;
};

} // namespace brain19
