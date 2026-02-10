#include "epistemic_promotion.hpp"
#include <algorithm>
#include <unordered_set>

namespace brain19 {

EpistemicPromotion::EpistemicPromotion(LongTermMemory& ltm)
    : ltm_(ltm)
{
}

std::vector<EpistemicPromotion::PromotionCandidate> EpistemicPromotion::evaluate_all() {
    std::vector<PromotionCandidate> candidates;

    auto all_ids = ltm_.get_active_concepts();
    for (auto id : all_ids) {
        auto candidate = evaluate(id);
        if (candidate) {
            candidates.push_back(std::move(*candidate));
        }
    }

    return candidates;
}

std::optional<EpistemicPromotion::PromotionCandidate> EpistemicPromotion::evaluate(ConceptId id) {
    auto cinfo = ltm_.retrieve_concept(id);
    if (!cinfo || cinfo->epistemic.is_invalidated()) {
        return std::nullopt;
    }

    // Check demotion first (takes priority)
    auto demotion = check_demotion(id, *cinfo);
    if (demotion) return demotion;

    // Check promotion based on current type
    switch (cinfo->epistemic.type) {
        case EpistemicType::SPECULATION:
            return check_speculation_to_hypothesis(id, *cinfo);
        case EpistemicType::HYPOTHESIS:
            return check_hypothesis_to_theory(id, *cinfo);
        case EpistemicType::THEORY:
            return check_theory_to_fact(id, *cinfo);
        default:
            return std::nullopt;
    }
}

bool EpistemicPromotion::promote(ConceptId id, EpistemicType new_type, double new_trust) {
    // FACT promotion is NOT allowed through this method
    if (new_type == EpistemicType::FACT) {
        return false;
    }

    auto cinfo = ltm_.retrieve_concept(id);
    if (!cinfo) return false;

    EpistemicMetadata new_meta(new_type, EpistemicStatus::ACTIVE, new_trust);
    return ltm_.update_epistemic_metadata(id, new_meta);
}

bool EpistemicPromotion::demote(ConceptId id, EpistemicType new_type, double new_trust,
                                 const std::string& /*reason*/) {
    auto cinfo = ltm_.retrieve_concept(id);
    if (!cinfo) return false;

    EpistemicMetadata new_meta(new_type, EpistemicStatus::ACTIVE, new_trust);
    return ltm_.update_epistemic_metadata(id, new_meta);
}

bool EpistemicPromotion::confirm_as_fact(ConceptId id, double trust,
                                          const std::string& /*human_note*/) {
    auto cinfo = ltm_.retrieve_concept(id);
    if (!cinfo) return false;

    // Only THEORY can be promoted to FACT
    if (cinfo->epistemic.type != EpistemicType::THEORY) {
        return false;
    }

    double fact_trust = std::max(0.8, std::min(1.0, trust));
    EpistemicMetadata new_meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, fact_trust);
    return ltm_.update_epistemic_metadata(id, new_meta);
}

EpistemicPromotion::MaintenanceResult EpistemicPromotion::run_maintenance() {
    MaintenanceResult result{0, 0, 0, {}};

    auto candidates = evaluate_all();

    for (auto& candidate : candidates) {
        if (candidate.requires_human_review) {
            result.pending_human_review.push_back(std::move(candidate));
            continue;
        }

        // Determine if promotion or demotion
        bool is_demotion = false;
        auto cinfo = ltm_.retrieve_concept(candidate.id);
        if (cinfo) {
            // Compare types: demotion if proposed trust < current trust
            is_demotion = (candidate.proposed_trust < candidate.current_trust);
        }

        if (is_demotion) {
            if (demote(candidate.id, candidate.proposed_type,
                       candidate.proposed_trust, candidate.reasoning)) {
                ++result.demotions;
            }
        } else {
            if (promote(candidate.id, candidate.proposed_type,
                        candidate.proposed_trust)) {
                ++result.promotions;
            }
        }
    }

    return result;
}

// ─── Private helpers ─────────────────────────────────────────────────────────

size_t EpistemicPromotion::count_supporting_relations(ConceptId id) const {
    auto incoming = ltm_.get_incoming_relations(id);
    size_t count = 0;
    for (const auto& rel : incoming) {
        if (rel.type == RelationType::SUPPORTS) {
            auto src = ltm_.retrieve_concept(rel.source);
            if (src && src->epistemic.is_active()) {
                ++count;
            }
        }
    }
    return count;
}

size_t EpistemicPromotion::count_supporting_from_type(
    ConceptId id, EpistemicType min_type) const
{
    auto incoming = ltm_.get_incoming_relations(id);
    size_t count = 0;

    // Type ordering: SPECULATION < HYPOTHESIS < THEORY < FACT
    auto type_rank = [](EpistemicType t) -> int {
        switch (t) {
            case EpistemicType::SPECULATION: return 0;
            case EpistemicType::HYPOTHESIS: return 1;
            case EpistemicType::INFERENCE: return 2;
            case EpistemicType::THEORY: return 3;
            case EpistemicType::DEFINITION: return 4;
            case EpistemicType::FACT: return 5;
            default: return -1;
        }
    };

    int min_rank = type_rank(min_type);

    for (const auto& rel : incoming) {
        if (rel.type == RelationType::SUPPORTS) {
            auto src = ltm_.retrieve_concept(rel.source);
            if (src && src->epistemic.is_active() &&
                type_rank(src->epistemic.type) >= min_rank) {
                ++count;
            }
        }
    }
    return count;
}

bool EpistemicPromotion::has_contradictions(ConceptId id) const {
    auto incoming = ltm_.get_incoming_relations(id);
    auto outgoing = ltm_.get_outgoing_relations(id);

    for (const auto& rel : incoming) {
        if (rel.type == RelationType::CONTRADICTS) {
            auto src = ltm_.retrieve_concept(rel.source);
            if (src && src->epistemic.is_active()) return true;
        }
    }
    for (const auto& rel : outgoing) {
        if (rel.type == RelationType::CONTRADICTS) {
            auto tgt = ltm_.retrieve_concept(rel.target);
            if (tgt && tgt->epistemic.is_active()) return true;
        }
    }
    return false;
}

size_t EpistemicPromotion::count_independent_sources(ConceptId id) const {
    auto incoming = ltm_.get_incoming_relations(id);
    std::unordered_set<ConceptId> sources;

    for (const auto& rel : incoming) {
        if (rel.type == RelationType::SUPPORTS) {
            auto src = ltm_.retrieve_concept(rel.source);
            if (src && src->epistemic.is_active()) {
                sources.insert(rel.source);
            }
        }
    }

    // Each unique source concept counts as independent
    return sources.size();
}

double EpistemicPromotion::compute_validation_score(ConceptId id) const {
    auto incoming = ltm_.get_incoming_relations(id);
    if (incoming.empty()) return 0.0;

    double total_weight = 0.0;
    double total_trust = 0.0;

    for (const auto& rel : incoming) {
        if (rel.type == RelationType::SUPPORTS) {
            auto src = ltm_.retrieve_concept(rel.source);
            if (src && src->epistemic.is_active()) {
                total_weight += rel.weight;
                total_trust += src->epistemic.trust * rel.weight;
            }
        }
    }

    if (total_weight < 0.001) return 0.0;
    return total_trust / total_weight;
}

std::optional<EpistemicPromotion::PromotionCandidate>
EpistemicPromotion::check_speculation_to_hypothesis(ConceptId id, const ConceptInfo& info) {
    // Criteria:
    // - At least 3 supporting relations
    // - Validation score > 0.3
    // - No contradictions

    size_t support_count = count_supporting_relations(id);
    double validation = compute_validation_score(id);
    bool contradictions = has_contradictions(id);

    if (support_count >= 3 && validation > 0.3 && !contradictions) {
        PromotionCandidate candidate;
        candidate.id = id;
        candidate.current_type = EpistemicType::SPECULATION;
        candidate.proposed_type = EpistemicType::HYPOTHESIS;
        candidate.current_trust = info.epistemic.trust;
        candidate.proposed_trust = std::min(0.5, std::max(0.3, validation));
        candidate.reasoning = "Met SPECULATION→HYPOTHESIS criteria: " +
                              std::to_string(support_count) + " supports, " +
                              "validation=" + std::to_string(validation) +
                              ", no contradictions";
        candidate.requires_human_review = false;

        // Collect evidence
        auto incoming = ltm_.get_incoming_relations(id);
        for (const auto& rel : incoming) {
            if (rel.type == RelationType::SUPPORTS) {
                candidate.evidence.push_back(rel.source);
            }
        }

        return candidate;
    }

    return std::nullopt;
}

std::optional<EpistemicPromotion::PromotionCandidate>
EpistemicPromotion::check_hypothesis_to_theory(ConceptId id, const ConceptInfo& info) {
    // Criteria:
    // - At least 5 supporting relations from THEORY+ concepts
    // - Validation score > 0.6
    // - At least 2 independent evidence sources
    // - No unresolved contradictions

    size_t theory_supports = count_supporting_from_type(id, EpistemicType::THEORY);
    double validation = compute_validation_score(id);
    size_t independent = count_independent_sources(id);
    bool contradictions = has_contradictions(id);

    if (theory_supports >= 5 && validation > 0.6 &&
        independent >= 2 && !contradictions) {

        PromotionCandidate candidate;
        candidate.id = id;
        candidate.current_type = EpistemicType::HYPOTHESIS;
        candidate.proposed_type = EpistemicType::THEORY;
        candidate.current_trust = info.epistemic.trust;
        candidate.proposed_trust = std::min(0.8, std::max(0.5, validation));
        candidate.reasoning = "Met HYPOTHESIS→THEORY criteria: " +
                              std::to_string(theory_supports) + " theory+ supports, " +
                              std::to_string(independent) + " independent sources, " +
                              "validation=" + std::to_string(validation);
        candidate.requires_human_review = false;

        auto incoming = ltm_.get_incoming_relations(id);
        for (const auto& rel : incoming) {
            if (rel.type == RelationType::SUPPORTS) {
                candidate.evidence.push_back(rel.source);
            }
        }

        return candidate;
    }

    return std::nullopt;
}

std::optional<EpistemicPromotion::PromotionCandidate>
EpistemicPromotion::check_theory_to_fact(ConceptId id, const ConceptInfo& info) {
    // THEORY → FACT ALWAYS requires human review
    // We just check if it's a reasonable candidate

    double validation = compute_validation_score(id);
    size_t support_count = count_supporting_relations(id);

    if (validation > 0.7 && support_count >= 5 && !has_contradictions(id)) {
        PromotionCandidate candidate;
        candidate.id = id;
        candidate.current_type = EpistemicType::THEORY;
        candidate.proposed_type = EpistemicType::FACT;
        candidate.current_trust = info.epistemic.trust;
        candidate.proposed_trust = std::max(0.8, validation);
        candidate.reasoning = "THEORY→FACT candidate (REQUIRES HUMAN REVIEW): " +
                              std::to_string(support_count) + " supports, " +
                              "validation=" + std::to_string(validation);
        candidate.requires_human_review = true;

        auto incoming = ltm_.get_incoming_relations(id);
        for (const auto& rel : incoming) {
            if (rel.type == RelationType::SUPPORTS) {
                candidate.evidence.push_back(rel.source);
            }
        }

        return candidate;
    }

    return std::nullopt;
}

std::optional<EpistemicPromotion::PromotionCandidate>
EpistemicPromotion::check_demotion(ConceptId id, const ConceptInfo& info) {
    // Check for contradictions → demote
    if (has_contradictions(id)) {
        EpistemicType demoted_type = info.epistemic.type;
        double demoted_trust = info.epistemic.trust;

        switch (info.epistemic.type) {
            case EpistemicType::FACT:
                demoted_type = EpistemicType::THEORY;
                demoted_trust = 0.6;
                break;
            case EpistemicType::THEORY:
                demoted_type = EpistemicType::HYPOTHESIS;
                demoted_trust = 0.4;
                break;
            case EpistemicType::HYPOTHESIS:
                demoted_type = EpistemicType::SPECULATION;
                demoted_trust = 0.2;
                break;
            default:
                // Already at lowest, just reduce trust
                demoted_trust = std::max(0.05, info.epistemic.trust - 0.1);
                break;
        }

        PromotionCandidate candidate;
        candidate.id = id;
        candidate.current_type = info.epistemic.type;
        candidate.proposed_type = demoted_type;
        candidate.current_trust = info.epistemic.trust;
        candidate.proposed_trust = demoted_trust;
        candidate.reasoning = "Demotion due to active contradiction";
        candidate.requires_human_review = false;
        return candidate;
    }

    return std::nullopt;
}

} // namespace brain19
