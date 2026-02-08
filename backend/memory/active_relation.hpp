#pragma once

#include <cstdint>
#include <chrono>

namespace brain19 {

using ConceptId = uint64_t;

// Relation types (must match LTM definition)
enum class RelationType {
    IS_A,
    HAS_PROPERTY,
    CAUSES,
    ENABLES,
    PART_OF,
    SIMILAR_TO,
    CONTRADICTS,
    SUPPORTS,
    TEMPORAL_BEFORE,
    CUSTOM
};

// Active relation in STM
// INVARIANT: STM stores ONLY activation, never relation content
struct ActiveRelation {
    ConceptId source;
    ConceptId target;
    RelationType type;
    double activation;  // [0.0, 1.0]
    std::chrono::steady_clock::time_point last_used;
    
    ActiveRelation()
        : source(0)
        , target(0)
        , type(RelationType::CUSTOM)
        , activation(0.0)
        , last_used(std::chrono::steady_clock::now())
    {}
    
    ActiveRelation(ConceptId src, ConceptId tgt, RelationType rel_type, double act)
        : source(src)
        , target(tgt)
        , type(rel_type)
        , activation(act)
        , last_used(std::chrono::steady_clock::now())
    {}
};

} // namespace brain19
