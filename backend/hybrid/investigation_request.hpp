#pragma once

#include "../common/types.hpp"
#include "../ltm/relation.hpp"
#include <string>
#include <cstdint>

namespace brain19 {

// =============================================================================
// ANOMALY TYPE
// =============================================================================

enum class AnomalyType {
    WEAK_EDGE,       // LTM weight low, KAN predicts strong
    CONTRADICTION,   // LTM and KAN disagree significantly
    MISSING_LINK,    // No LTM relation, but KAN predicts one
    STALE_RELATION   // LTM weight high, KAN predicts weak
};

inline const char* anomaly_type_to_string(AnomalyType t) {
    switch (t) {
        case AnomalyType::WEAK_EDGE:     return "WEAK_EDGE";
        case AnomalyType::CONTRADICTION:  return "CONTRADICTION";
        case AnomalyType::MISSING_LINK:   return "MISSING_LINK";
        case AnomalyType::STALE_RELATION: return "STALE_RELATION";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// INVESTIGATION REQUEST
// =============================================================================
//
// Bridge between KAN anomaly detection (KanGraphMonitor) and
// MiniLLM investigation (KanAwareMiniLLM::investigate_anomalies).
//
// CRITICAL:
// - This is a signal, not knowledge
// - No epistemic metadata (not stored in LTM)
// - READ-ONLY references to concepts
//

struct InvestigationRequest {
    uint64_t request_id;
    AnomalyType anomaly_type;
    ConceptId concept_a;
    ConceptId concept_b;
    RelationType relation_type;
    double ltm_weight;         // 0.0 if no relation exists
    double kan_score;
    double anomaly_strength;   // |ltm - kan| or kan for missing links
    std::string description;

    // No default constructor
    InvestigationRequest() = delete;

    InvestigationRequest(
        uint64_t id,
        AnomalyType type,
        ConceptId a,
        ConceptId b,
        RelationType rel_type,
        double ltm_w,
        double kan_s,
        double strength,
        std::string desc
    ) : request_id(id)
      , anomaly_type(type)
      , concept_a(a)
      , concept_b(b)
      , relation_type(rel_type)
      , ltm_weight(ltm_w)
      , kan_score(kan_s)
      , anomaly_strength(strength)
      , description(std::move(desc))
    {}
};

} // namespace brain19
