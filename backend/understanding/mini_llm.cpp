#include "mini_llm.hpp"
#include <sstream>

namespace brain19 {

// =============================================================================
// STUB MINI-LLM IMPLEMENTATION
// =============================================================================

std::vector<MeaningProposal> StubMiniLLM::extract_meaning(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<MeaningProposal> proposals;

    if (active_concepts.empty()) {
        return proposals;
    }

    // CRITICAL: READ-ONLY access to LTM
    // Generate simple meaning proposal based on active concepts

    std::ostringstream interpretation;
    interpretation << "Detected activation pattern involving " << active_concepts.size() << " concepts";

    std::ostringstream reasoning;
    reasoning << "Stub Mini-LLM detected co-activation of concepts: ";
    for (size_t i = 0; i < active_concepts.size() && i < 3; ++i) {
        auto concept_info = ltm.retrieve_concept(active_concepts[i]);  // READ-ONLY
        if (concept_info.has_value()) {
            reasoning << concept_info->label;
            if (i < active_concepts.size() - 1 && i < 2) {
                reasoning << ", ";
            }
        }
    }
    if (active_concepts.size() > 3) {
        reasoning << " and " << (active_concepts.size() - 3) << " more";
    }

    proposals.emplace_back(
        ++proposal_counter_,
        active_concepts,
        interpretation.str(),
        reasoning.str(),
        0.5,  // Conservative confidence
        get_model_id()
    );

    // VERIFICATION: Check that epistemic_type is HYPOTHESIS
    // This is enforced by MeaningProposal constructor
    if (!proposals.empty()) {
        if (proposals[0].epistemic_type != EpistemicType::HYPOTHESIS) {
            // CRITICAL ERROR: Should be impossible
            throw std::logic_error("EPISTEMIC VIOLATION: MeaningProposal not HYPOTHESIS");
        }
    }

    return proposals;
}

std::vector<HypothesisProposal> StubMiniLLM::generate_hypotheses(
    const std::vector<ConceptId>& evidence_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<HypothesisProposal> proposals;

    if (evidence_concepts.size() < 2) {
        return proposals;  // Need at least 2 concepts for pattern
    }

    // CRITICAL: READ-ONLY access to LTM
    std::ostringstream statement;
    statement << "Hypothetical relationship detected between ";

    std::vector<std::string> concept_labels;
    for (size_t i = 0; i < evidence_concepts.size() && i < 3; ++i) {
        auto concept_info = ltm.retrieve_concept(evidence_concepts[i]);  // READ-ONLY
        if (concept_info.has_value()) {
            concept_labels.push_back(concept_info->label);
        }
    }

    if (concept_labels.size() >= 2) {
        statement << concept_labels[0] << " and " << concept_labels[1];
        if (concept_labels.size() > 2) {
            statement << " and others";
        }
    }

    std::ostringstream reasoning;
    reasoning << "Stub Mini-LLM detected co-occurrence pattern";

    std::vector<std::string> patterns = {
        "co-activation",
        "temporal correlation"
    };

    proposals.emplace_back(
        ++proposal_counter_,
        evidence_concepts,
        statement.str(),
        reasoning.str(),
        patterns,
        0.4,  // Very conservative confidence
        get_model_id()
    );

    // VERIFICATION: Check that suggested_epistemic.suggested_type is HYPOTHESIS
    if (!proposals.empty()) {
        if (proposals[0].suggested_epistemic.suggested_type != EpistemicType::HYPOTHESIS) {
            // CRITICAL ERROR: Should be impossible
            throw std::logic_error("EPISTEMIC VIOLATION: HypothesisProposal not HYPOTHESIS");
        }
    }

    return proposals;
}

std::vector<AnalogyProposal> StubMiniLLM::detect_analogies(
    const std::vector<ConceptId>& concept_set_a,
    const std::vector<ConceptId>& concept_set_b,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<AnalogyProposal> proposals;

    if (concept_set_a.empty() || concept_set_b.empty()) {
        return proposals;
    }

    // CRITICAL: READ-ONLY access to LTM
    std::ostringstream mapping;
    mapping << "Structural similarity detected between "
            << concept_set_a.size() << " concepts in domain A and "
            << concept_set_b.size() << " concepts in domain B";

    // Compute simple structural similarity (stub: just size similarity)
    double size_ratio = static_cast<double>(std::min(concept_set_a.size(), concept_set_b.size())) /
                        static_cast<double>(std::max(concept_set_a.size(), concept_set_b.size()));

    proposals.emplace_back(
        ++proposal_counter_,
        concept_set_a,
        concept_set_b,
        mapping.str(),
        size_ratio * 0.5,  // Conservative structural similarity
        0.3,  // Low confidence (stub)
        get_model_id()
    );

    return proposals;
}

std::vector<ContradictionProposal> StubMiniLLM::detect_contradictions(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) const {
    std::vector<ContradictionProposal> proposals;

    if (active_concepts.size() < 2) {
        return proposals;  // Need at least 2 concepts for contradiction
    }

    // CRITICAL: READ-ONLY access to LTM
    // Stub: Check if any concepts have conflicting epistemic statuses
    for (size_t i = 0; i < active_concepts.size() && i < 5; ++i) {
        for (size_t j = i + 1; j < active_concepts.size() && j < 5; ++j) {
            auto concept_a = ltm.retrieve_concept(active_concepts[i]);  // READ-ONLY
            auto concept_b = ltm.retrieve_concept(active_concepts[j]);  // READ-ONLY

            if (!concept_a.has_value() || !concept_b.has_value()) {
                continue;
            }

            // Stub logic: Mark if one is INVALIDATED and other is ACTIVE
            if (concept_a->epistemic.status == EpistemicStatus::INVALIDATED &&
                concept_b->epistemic.status == EpistemicStatus::ACTIVE) {

                std::ostringstream description;
                description << "Potential inconsistency: " << concept_a->label
                           << " (INVALIDATED) co-active with " << concept_b->label
                           << " (ACTIVE)";

                proposals.emplace_back(
                    ++proposal_counter_,
                    active_concepts[i],
                    active_concepts[j],
                    description.str(),
                    "Stub Mini-LLM detected epistemic status mismatch",
                    0.6,  // Moderate severity
                    0.5,  // Moderate confidence
                    get_model_id()
                );
            }
        }
    }

    return proposals;
}

} // namespace brain19
