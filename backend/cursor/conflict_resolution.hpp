#pragma once

#include "../ltm/long_term_memory.hpp"

namespace brain19 {

// =============================================================================
// CONFLICT RESOLUTION
// =============================================================================
//
// Computes effective priority for a concept using a weighted combination
// of structural confidence, semantic confidence, and activation score.
//
// Formula:
//   effective_priority = alpha * structural_confidence
//                      + beta  * semantic_confidence
//                      + gamma * activation_score
//
// Used to resolve conflicts when multiple concepts compete for attention
// or when contradicting concepts are activated simultaneously.
//

struct ConflictWeights {
    double alpha = 0.4;  // structural confidence weight
    double beta  = 0.4;  // semantic confidence weight
    double gamma = 0.2;  // activation weight
};

inline double effective_priority(const ConceptInfo& c, const ConflictWeights& w = {}) {
    return w.alpha * c.structural_confidence
         + w.beta  * c.semantic_confidence
         + w.gamma * c.activation_score;
}

// Resolve conflict between two concepts: returns true if a wins over b
inline bool resolves_in_favor(const ConceptInfo& a, const ConceptInfo& b,
                               const ConflictWeights& w = {}) {
    return effective_priority(a, w) > effective_priority(b, w);
}

} // namespace brain19
