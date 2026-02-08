#pragma once

#include "activation_level.hpp"
#include <cstdint>
#include <chrono>

namespace brain19 {

using ConceptId = uint64_t;

// STM entry for an activated concept
// INVARIANT: STM stores ONLY activation, never knowledge content
struct STMEntry {
    ConceptId concept_id;
    double activation;  // [0.0, 1.0]
    ActivationClass classification;
    std::chrono::steady_clock::time_point last_used;
    
    STMEntry()
        : concept_id(0)
        , activation(0.0)
        , classification(ActivationClass::CONTEXTUAL)
        , last_used(std::chrono::steady_clock::now())
    {}
    
    STMEntry(ConceptId id, double act, ActivationClass cls)
        : concept_id(id)
        , activation(act)
        , classification(cls)
        , last_used(std::chrono::steady_clock::now())
    {}
};

} // namespace brain19
