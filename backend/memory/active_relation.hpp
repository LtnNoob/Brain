#pragma once

#include <cstdint>
#include <chrono>
#include "../common/types.hpp"

namespace brain19 {


// Relation types — widened to uint16_t for runtime extensibility.
// Original 10 values (0-9) are unchanged for binary compatibility.
// New built-in types: 10-19. Runtime-registered types: >= 1000.
enum class RelationType : uint16_t {
    // === Original 10 (0-9) — backward compatible ===
    IS_A            = 0,
    HAS_PROPERTY    = 1,
    CAUSES          = 2,
    ENABLES         = 3,
    PART_OF         = 4,
    SIMILAR_TO      = 5,
    CONTRADICTS     = 6,
    SUPPORTS        = 7,
    TEMPORAL_BEFORE = 8,
    CUSTOM          = 9,

    // === New built-in types (10-19) ===
    PRODUCES        = 10,
    REQUIRES        = 11,
    USES            = 12,
    SOURCE          = 13,
    HAS_PART        = 14,   // inverse of PART_OF
    TEMPORAL_AFTER  = 15,   // inverse of TEMPORAL_BEFORE
    INSTANCE_OF     = 16,
    DERIVED_FROM    = 17,
    IMPLIES         = 18,
    ASSOCIATED_WITH = 19,

    // === Linguistic relation types (20-28) ===
    SUBJECT_OF      = 20,   // Word → Sentence: word is subject of sentence
    OBJECT_OF       = 21,   // Word → Sentence: word is object of sentence
    VERB_OF         = 22,   // Word → Sentence: word is verb of sentence
    MODIFIER_OF     = 23,   // Word → Word/Sentence: modifier relationship
    DENOTES         = 24,   // Word-Concept → Semantic Concept
    PART_OF_SENTENCE= 25,   // Generic: word belongs to sentence (fallback)
    TEMPORAL_OF     = 26,   // Time expression → Sentence
    LOCATIVE_OF     = 27,   // Location expression → Sentence
    PRECEDES        = 28,   // Sentence → Sentence (discourse order)

    // === Runtime-registered types start here ===
    RUNTIME_BASE    = 1000,
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
