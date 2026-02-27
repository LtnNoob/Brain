#pragma once

#include "../memory/relation_type_registry.hpp"
#include <unordered_map>

namespace brain19 {

// =============================================================================
// RELATION BEHAVIOR CONFIG
// =============================================================================
//
// Central configuration governing how each RelationCategory behaves across
// ALL subsystems. No subsystem may treat relations uniformly.
//
// Consumers:
//   - Spreading Activation:    spreading_weight, spreading_direction
//   - Embedding Training:      embedding_alpha
//   - Property Inheritance:    trust_decay_per_hop, inherit_properties, inherit_dir
//   - ConceptModel Training:   category-based features
//   - KAN Decoder:             category-allocated output dimensions
//   - Sleep/REM:               category-filtered random walks
//   - Language Templates:      per-relation syntax
//   - Epistemic Demotion:      weighted contradiction ratio
//

enum class InheritDirection : uint8_t {
    NONE,       // No property inheritance across this relation
    FORWARD,    // Source → Target (e.g., IS_A: parent→child)
    REVERSE,    // Target → Source (e.g., PART_OF: part→whole)
    BOTH        // Bidirectional
};

struct RelationBehavior {
    float spreading_weight;      // Spreading Activation multiplier [0..1]
    float spreading_direction;   // +1.0 = excitatory, -1.0 = inhibitory
    float embedding_alpha;       // Embedding training: + = attract, - = repel
    float trust_decay_per_hop;   // Property inheritance decay per hop [0..1]
    bool  inherit_properties;    // Whether properties propagate across this relation
    InheritDirection inherit_dir;// Direction of property inheritance
};

// Lookup table: one entry per RelationCategory.
// Maps existing 9 categories to the behaviors from convergence-design-v2.
//
// Design mapping:
//   SIMILARITY   → ASSOCIATIVE role   (excitatory, moderate spread)
//   FUNCTIONAL   → DEPENDENCY role    (excitatory, strong spread, inheritable)
//   EPISTEMIC    → SUPPORTS-like      (excitatory, moderate spread)
//   CUSTOM_CATEGORY → safe defaults
//
inline const std::unordered_map<RelationCategory, RelationBehavior>& get_relation_behaviors() {
    static const std::unordered_map<RelationCategory, RelationBehavior> behaviors = {
        //                                        spread_w  dir    emb_α   decay  inherit  inherit_dir
        {RelationCategory::HIERARCHICAL,   {1.0f,  +1.0f,  0.30f,  0.90f, true,  InheritDirection::FORWARD}},
        {RelationCategory::COMPOSITIONAL,  {0.8f,  +1.0f,  0.05f,  1.00f, true,  InheritDirection::REVERSE}},   // Teil→Ganzes (0.05: HAS_PROPERTY shared props must not collapse embeddings)
        {RelationCategory::CAUSAL,         {0.7f,  +1.0f,  0.15f,  0.85f, false, InheritDirection::FORWARD}},
        {RelationCategory::SIMILARITY,     {0.6f,  +1.0f,  0.20f,  0.95f, false, InheritDirection::NONE}},      // ≈ ASSOCIATIVE
        {RelationCategory::OPPOSITION,     {0.5f,  -1.0f, -0.25f,  0.00f, false, InheritDirection::NONE}},      // INHIBITORY
        {RelationCategory::EPISTEMIC,      {0.7f,  +1.0f,  0.15f,  0.90f, false, InheritDirection::NONE}},      // SUPPORTS
        {RelationCategory::TEMPORAL,       {0.4f,  +1.0f,  0.10f,  0.90f, false, InheritDirection::FORWARD}},
        {RelationCategory::FUNCTIONAL,     {0.8f,  +1.0f,  0.20f,  0.95f, true,  InheritDirection::FORWARD}},   // ≈ DEPENDENCY
        {RelationCategory::CUSTOM_CATEGORY,{0.5f,  +1.0f,  0.10f,  0.90f, false, InheritDirection::NONE}},
    };
    return behaviors;
}

// Convenience: get behavior for a category (with safe fallback)
inline const RelationBehavior& get_behavior(RelationCategory cat) {
    static const RelationBehavior DEFAULT{0.5f, +1.0f, 0.10f, 0.90f, false, InheritDirection::NONE};
    const auto& behaviors = get_relation_behaviors();
    auto it = behaviors.find(cat);
    if (it != behaviors.end()) return it->second;
    return DEFAULT;
}

// Convenience: get behavior for a RelationType via registry lookup
inline const RelationBehavior& get_behavior(RelationType type) {
    RelationCategory cat = RelationTypeRegistry::instance().get_category(type);
    return get_behavior(cat);
}

} // namespace brain19
