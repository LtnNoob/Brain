#pragma once

#include "../common/types.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include "../ltm/long_term_memory.hpp"
#include <string>
#include <vector>
#include <optional>

namespace brain19 {

// =============================================================================
// EPISTEMIC PROMOTION
// =============================================================================
//
// Manages the lifecycle of epistemic status:
//   SPECULATION (0.1-0.3) → HYPOTHESIS (0.3-0.5) → THEORY (0.5-0.8) → FACT (0.8-1.0)
//
// CRITICAL INVARIANT:
// - THEORY → FACT requires human review (NEVER automatic)
// - Demotion can happen automatically on contradictions
// - Deprecated after 30 days of non-use
//

class EpistemicPromotion {
public:
    explicit EpistemicPromotion(LongTermMemory& ltm);

    struct PromotionCandidate {
        ConceptId id;
        EpistemicType current_type;
        EpistemicType proposed_type;
        double current_trust;
        double proposed_trust;
        std::string reasoning;
        std::vector<ConceptId> evidence;
        bool requires_human_review;  // true for THEORY→FACT
    };

    // Evaluate all concepts for promotion/demotion
    std::vector<PromotionCandidate> evaluate_all();

    // Evaluate single concept
    std::optional<PromotionCandidate> evaluate(ConceptId id);

    // Apply promotion (non-FACT only)
    bool promote(ConceptId id, EpistemicType new_type, double new_trust);

    // Apply demotion
    bool demote(ConceptId id, EpistemicType new_type, double new_trust,
                const std::string& reason);

    // Human confirms FACT promotion
    bool confirm_as_fact(ConceptId id, double trust, const std::string& human_note);

    // Automatic maintenance (run periodically)
    struct MaintenanceResult {
        size_t promotions;
        size_t demotions;
        size_t deprecations;
        std::vector<PromotionCandidate> pending_human_review;
    };
    MaintenanceResult run_maintenance();

private:
    LongTermMemory& ltm_;

    // Count supporting relations (SUPPORTS type) from active concepts
    size_t count_supporting_relations(ConceptId id) const;

    // Count supporting relations from concepts of given type or higher
    size_t count_supporting_from_type(ConceptId id, EpistemicType min_type) const;

    // Check if concept has contradictions
    bool has_contradictions(ConceptId id) const;

    // Count independent evidence sources
    size_t count_independent_sources(ConceptId id) const;

    // Compute synthetic validation score based on relation quality
    double compute_validation_score(ConceptId id) const;

    // Check promotion: SPECULATION → HYPOTHESIS
    std::optional<PromotionCandidate> check_speculation_to_hypothesis(
        ConceptId id, const ConceptInfo& info);

    // Check promotion: HYPOTHESIS → THEORY
    std::optional<PromotionCandidate> check_hypothesis_to_theory(
        ConceptId id, const ConceptInfo& info);

    // Check promotion: THEORY → FACT (always requires human)
    std::optional<PromotionCandidate> check_theory_to_fact(
        ConceptId id, const ConceptInfo& info);

    // Check for demotion conditions
    std::optional<PromotionCandidate> check_demotion(
        ConceptId id, const ConceptInfo& info);
};

} // namespace brain19
