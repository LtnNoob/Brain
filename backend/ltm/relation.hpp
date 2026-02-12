#pragma once

#include "../memory/active_relation.hpp"
#include "../memory/relation_type_registry.hpp"
#include <cstdint>
#include <string>
#include "../common/types.hpp"

namespace brain19 {


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

    // === Refactor: Dynamic fields for FocusCursor / Global Dynamics ===
    // Defaults = 0.0, carved from PersistentRelationRecord._reserved (24B)
    double dynamic_weight = 0.0;       // Runtime-adjusted weight [0,1]
    double inhibition_factor = 0.0;    // Inhibition from conflict [0,1]
    double structural_strength = 0.0;  // Graph-structural strength [0,1]

    // Constructor with validation (existing signature unchanged)
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
        // New fields initialized via default member initializers
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

// Convert RelationType to string — delegates to registry for dynamic type support
inline const char* relation_type_to_string(RelationType type) {
    return RelationTypeRegistry::instance().get_name(type).c_str();
}

} // namespace brain19
