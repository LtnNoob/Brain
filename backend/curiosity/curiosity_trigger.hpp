#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace brain19 {

using ContextId = uint64_t;
using ConceptId = uint64_t;

// Types of curiosity signals
enum class TriggerType {
    SHALLOW_RELATIONS,      // Concepts activated but few relations
    MISSING_DEPTH,          // Repeated patterns without deeper structure
    LOW_EXPLORATION,        // Stable context with little variation
    RECURRENT_WITHOUT_FUNCTION,  // Same activation pattern, no learned function
    UNKNOWN
};

// CuriosityTrigger: Pure data signal
// Contains NO logic, only observation metadata
struct CuriosityTrigger {
    TriggerType type;
    ContextId context_id;
    std::vector<ConceptId> related_concept_ids;
    std::string description;
    
    CuriosityTrigger()
        : type(TriggerType::UNKNOWN)
        , context_id(0)
    {}
    
    CuriosityTrigger(
        TriggerType t,
        ContextId ctx,
        const std::vector<ConceptId>& concepts,
        const std::string& desc
    )
        : type(t)
        , context_id(ctx)
        , related_concept_ids(concepts)
        , description(desc)
    {}
};

} // namespace brain19
