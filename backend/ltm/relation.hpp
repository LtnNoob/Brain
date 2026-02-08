#pragma once

#include "../memory/active_relation.hpp"
#include <cstdint>
#include <string>

namespace brain19 {

using ConceptId = uint64_t;
using RelationId = uint64_t;

// =============================================================================
// PERSISTENT RELATION
// =============================================================================
//
// RelationInfo: Persistent relation between concepts stored in LTM
//
// CRITICAL CONSTRAINTS:
// - Relations are DIRECTED (source -> target)
// - Weight MUST be in [0.0, 1.0]
// - Weight affects spreading activation strength
// - Relations are NOT epistemic entities (no trust/type)
// - Relations model semantic connectivity
//
// NOTE: This is ADDITIVE to the existing codebase.
// STM still uses ActiveRelation for short-term tracking.
// LTM uses RelationInfo for persistent storage.
//
struct RelationInfo {
    RelationId id;
    ConceptId source;
    ConceptId target;
    RelationType type;
    double weight;  // [0.0, 1.0] - spreading activation strength

    // Constructor with validation
    RelationInfo(
        RelationId relation_id,
        ConceptId src,
        ConceptId tgt,
        RelationType rel_type,
        double rel_weight
    ) : id(relation_id)
      , source(src)
      , target(tgt)
      , type(rel_type)
      , weight(clamp_weight(rel_weight))
    {
    }

    // Default constructor deleted - force explicit construction
    RelationInfo() = delete;

    // Copy/move allowed
    RelationInfo(const RelationInfo&) = default;
    RelationInfo(RelationInfo&&) = default;
    RelationInfo& operator=(const RelationInfo&) = default;
    RelationInfo& operator=(RelationInfo&&) = default;

private:
    static double clamp_weight(double w) {
        if (w < 0.0) return 0.0;
        if (w > 1.0) return 1.0;
        return w;
    }
};

// Convert RelationType to string (mirrors active_relation.hpp)
inline const char* relation_type_to_string(RelationType type) {
    switch (type) {
        case RelationType::IS_A: return "IS_A";
        case RelationType::HAS_PROPERTY: return "HAS_PROPERTY";
        case RelationType::CAUSES: return "CAUSES";
        case RelationType::ENABLES: return "ENABLES";
        case RelationType::PART_OF: return "PART_OF";
        case RelationType::SIMILAR_TO: return "SIMILAR_TO";
        case RelationType::CONTRADICTS: return "CONTRADICTS";
        case RelationType::SUPPORTS: return "SUPPORTS";
        case RelationType::TEMPORAL_BEFORE: return "TEMPORAL_BEFORE";
        case RelationType::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

} // namespace brain19
