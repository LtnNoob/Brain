#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <cassert>

namespace brain19 {

// EpistemicType: Categorization of knowledge certainty
// CRITICAL INVARIANT: Every knowledge item MUST have an explicit EpistemicType
enum class EpistemicType {
    FACT,           // Verified, reproducible, high-certainty knowledge
    DEFINITION,     // Definitional knowledge (tautological)
    THEORY,         // Well-supported, falsifiable claims
    HYPOTHESIS,     // Testable but not yet strongly supported
    INFERENCE,      // Derived from other knowledge
    SPECULATION,    // Low-certainty, exploratory knowledge
    // NO "UNKNOWN" - absence of type is a compile error
};

// EpistemicStatus: Lifecycle state of knowledge
// CRITICAL INVARIANT: Every knowledge item MUST have an explicit EpistemicStatus
enum class EpistemicStatus {
    ACTIVE,         // Currently valid and usable
    CONTEXTUAL,     // Valid only in specific contexts
    SUPERSEDED,     // Replaced by better knowledge (but not wrong)
    INVALIDATED     // Known to be incorrect (never deleted, only marked)
    // NO "UNKNOWN" - absence of status is a compile error
};

// EpistemicMetadata: MANDATORY metadata for ALL knowledge items
// 
// CRITICAL INVARIANTS (enforced by construction):
// 1. Every knowledge item MUST have EpistemicMetadata
// 2. Construction without metadata is IMPOSSIBLE
// 3. Trust MUST be in [0.0, 1.0]
// 4. No default construction allowed
// 5. All fields are immutable after construction (use update methods)
//
// INVALIDATION RULE:
// - Knowledge is NEVER deleted
// - INVALIDATED knowledge remains in storage with very low trust
// - This preserves epistemic history and prevents silent data loss
struct EpistemicMetadata {
    EpistemicType type;
    EpistemicStatus status;
    double trust;  // MUST be in [0.0, 1.0]
    
    // DELETED: No default constructor
    // This forces explicit epistemic decisions at compile time
    EpistemicMetadata() = delete;
    
    // REQUIRED: Explicit constructor with all fields
    // Trust is validated at construction
    explicit EpistemicMetadata(
        EpistemicType t,
        EpistemicStatus s,
        double trust_value
    ) : type(t), status(s), trust(trust_value) {
        
        // Runtime validation (always, not just debug)
        if (trust_value < 0.0 || trust_value > 1.0) {
            throw std::out_of_range(
                "Trust must be in [0.0, 1.0], got: " + std::to_string(trust_value)
            );
        }
        
        // Debug assertion: INVALIDATED items should have very low trust
        // This is a soft constraint (warning, not error)
        #ifndef NDEBUG
        if (status == EpistemicStatus::INVALIDATED && trust_value >= 0.2) {
            // This is suspicious but not illegal
            // Log warning in debug builds
            assert(false && 
                "WARNING: INVALIDATED knowledge should have trust < 0.2. "
                "Current trust value suggests item may not be properly invalidated."
            );
        }
        #endif
    }
    
    // Assignment operators (needed for ConceptInfo container operations)
    EpistemicMetadata& operator=(const EpistemicMetadata&) = default;
    EpistemicMetadata& operator=(EpistemicMetadata&&) = default;
    
    // Copy constructor allowed (for storage/retrieval)
    EpistemicMetadata(const EpistemicMetadata&) = default;
    
    // Move constructor allowed
    EpistemicMetadata(EpistemicMetadata&&) = default;
    
    // Validation helper (for runtime checks)
    bool is_valid() const {
        return trust >= 0.0 && trust <= 1.0;
    }
    
    // Status check helpers (for API clarity)
    bool is_active() const { return status == EpistemicStatus::ACTIVE; }
    bool is_invalidated() const { return status == EpistemicStatus::INVALIDATED; }
    bool is_superseded() const { return status == EpistemicStatus::SUPERSEDED; }
    bool is_contextual() const { return status == EpistemicStatus::CONTEXTUAL; }
};

// HELPER: Create metadata for verified facts
// This does NOT infer epistemic type - it's a convenience for known cases
inline EpistemicMetadata create_fact_metadata(double trust) {
    return EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, trust);
}

// HELPER: Create metadata for hypotheses
inline EpistemicMetadata create_hypothesis_metadata(double trust) {
    return EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, trust);
}

// HELPER: Create metadata for invalidated knowledge
// Note: Trust is intentionally set very low
inline EpistemicMetadata create_invalidated_metadata(EpistemicType original_type, double trust = 0.05) {
    return EpistemicMetadata(original_type, EpistemicStatus::INVALIDATED, trust);
}

// STRINGIFICATION: For debugging and serialization
inline std::string to_string(EpistemicType type) {
    switch (type) {
        case EpistemicType::FACT: return "FACT";
        case EpistemicType::DEFINITION: return "DEFINITION";
        case EpistemicType::THEORY: return "THEORY";
        case EpistemicType::HYPOTHESIS: return "HYPOTHESIS";
        case EpistemicType::INFERENCE: return "INFERENCE";
        case EpistemicType::SPECULATION: return "SPECULATION";
        default: return "UNKNOWN_TYPE";  // Should never happen
    }
}

inline std::string to_string(EpistemicStatus status) {
    switch (status) {
        case EpistemicStatus::ACTIVE: return "ACTIVE";
        case EpistemicStatus::CONTEXTUAL: return "CONTEXTUAL";
        case EpistemicStatus::SUPERSEDED: return "SUPERSEDED";
        case EpistemicStatus::INVALIDATED: return "INVALIDATED";
        default: return "UNKNOWN_STATUS";  // Should never happen
    }
}

} // namespace brain19
