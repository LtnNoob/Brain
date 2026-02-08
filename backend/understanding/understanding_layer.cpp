#include "understanding_layer.hpp"
#include <iostream>
#include <algorithm>

namespace brain19 {

// =============================================================================
// CONSTRUCTOR / DESTRUCTOR
// =============================================================================

UnderstandingLayer::UnderstandingLayer(UnderstandingLayerConfig config)
    : config_(config)
    , stats_()
{
    log_message("UnderstandingLayer initialized");
}

UnderstandingLayer::~UnderstandingLayer() {
    log_message("UnderstandingLayer destroyed");
}

// =============================================================================
// MINI-LLM MANAGEMENT
// =============================================================================

void UnderstandingLayer::register_mini_llm(std::unique_ptr<MiniLLM> mini_llm) {
    if (!mini_llm) {
        throw std::invalid_argument("Cannot register null Mini-LLM");
    }

    log_message("Registered Mini-LLM: " + mini_llm->get_model_id());
    mini_llms_.push_back(std::move(mini_llm));
}

// =============================================================================
// SEMANTIC ANALYSIS
// =============================================================================

std::vector<MeaningProposal> UnderstandingLayer::analyze_meaning(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) {
    std::vector<MeaningProposal> all_proposals;

    if (mini_llms_.empty()) {
        log_message("Warning: No Mini-LLMs registered");
        return all_proposals;
    }

    // Collect proposals from all Mini-LLMs
    for (const auto& mini_llm : mini_llms_) {
        auto proposals = mini_llm->extract_meaning(active_concepts, ltm, stm, context);

        // CRITICAL: Verify all proposals are HYPOTHESIS
        for (const auto& proposal : proposals) {
            if (proposal.epistemic_type != EpistemicType::HYPOTHESIS) {
                throw std::logic_error(
                    "EPISTEMIC VIOLATION: MeaningProposal from " +
                    mini_llm->get_model_id() + " is not HYPOTHESIS"
                );
            }
        }

        all_proposals.insert(all_proposals.end(), proposals.begin(), proposals.end());
    }

    // Filter by confidence threshold
    auto filtered = filter_proposals_by_confidence(all_proposals, config_.min_meaning_confidence);

    stats_.total_meaning_proposals += filtered.size();
    log_message("Generated " + std::to_string(filtered.size()) + " meaning proposals");

    return filtered;
}

std::vector<HypothesisProposal> UnderstandingLayer::propose_hypotheses(
    const std::vector<ConceptId>& evidence_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) {
    std::vector<HypothesisProposal> all_proposals;

    if (mini_llms_.empty()) {
        log_message("Warning: No Mini-LLMs registered");
        return all_proposals;
    }

    // Collect proposals from all Mini-LLMs
    for (const auto& mini_llm : mini_llms_) {
        auto proposals = mini_llm->generate_hypotheses(evidence_concepts, ltm, stm, context);

        // CRITICAL: Verify all proposals suggest HYPOTHESIS
        for (const auto& proposal : proposals) {
            if (proposal.suggested_epistemic.suggested_type != EpistemicType::HYPOTHESIS) {
                throw std::logic_error(
                    "EPISTEMIC VIOLATION: HypothesisProposal from " +
                    mini_llm->get_model_id() + " does not suggest HYPOTHESIS"
                );
            }
        }

        all_proposals.insert(all_proposals.end(), proposals.begin(), proposals.end());
    }

    // Filter by confidence threshold
    auto filtered = filter_proposals_by_confidence(all_proposals, config_.min_hypothesis_confidence);

    stats_.total_hypothesis_proposals += filtered.size();
    log_message("Generated " + std::to_string(filtered.size()) + " hypothesis proposals");

    return filtered;
}

std::vector<AnalogyProposal> UnderstandingLayer::find_analogies(
    const std::vector<ConceptId>& concept_set_a,
    const std::vector<ConceptId>& concept_set_b,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) {
    std::vector<AnalogyProposal> all_proposals;

    if (mini_llms_.empty()) {
        log_message("Warning: No Mini-LLMs registered");
        return all_proposals;
    }

    // Collect proposals from all Mini-LLMs
    for (const auto& mini_llm : mini_llms_) {
        auto proposals = mini_llm->detect_analogies(concept_set_a, concept_set_b, ltm, stm, context);
        all_proposals.insert(all_proposals.end(), proposals.begin(), proposals.end());
    }

    // Filter by confidence threshold
    auto filtered = filter_proposals_by_confidence(all_proposals, config_.min_analogy_confidence);

    stats_.total_analogy_proposals += filtered.size();
    log_message("Generated " + std::to_string(filtered.size()) + " analogy proposals");

    return filtered;
}

std::vector<ContradictionProposal> UnderstandingLayer::check_contradictions(
    const std::vector<ConceptId>& active_concepts,
    const LongTermMemory& ltm,
    const ShortTermMemory& stm,
    ContextId context
) {
    std::vector<ContradictionProposal> all_proposals;

    if (mini_llms_.empty()) {
        log_message("Warning: No Mini-LLMs registered");
        return all_proposals;
    }

    // Collect proposals from all Mini-LLMs
    for (const auto& mini_llm : mini_llms_) {
        auto proposals = mini_llm->detect_contradictions(active_concepts, ltm, stm, context);
        all_proposals.insert(all_proposals.end(), proposals.begin(), proposals.end());
    }

    // Filter by severity threshold
    std::vector<ContradictionProposal> filtered;
    for (const auto& proposal : all_proposals) {
        if (proposal.severity >= config_.min_contradiction_severity) {
            filtered.push_back(proposal);
        }
    }

    stats_.total_contradiction_proposals += filtered.size();
    log_message("Generated " + std::to_string(filtered.size()) + " contradiction proposals");

    return filtered;
}

// =============================================================================
// INTEGRATED ANALYSIS
// =============================================================================

UnderstandingLayer::UnderstandingResult UnderstandingLayer::perform_understanding_cycle(
    ConceptId seed_concept,
    CognitiveDynamics& cognitive_dynamics,
    const LongTermMemory& ltm,
    ShortTermMemory& stm,
    ContextId context
) {
    UnderstandingResult result;

    log_message("Starting understanding cycle from seed concept " + std::to_string(seed_concept));

    // PHASE 1: Use Cognitive Dynamics for spreading activation
    auto spread_stats = cognitive_dynamics.spread_activation(seed_concept, 1.0, context, ltm, stm);

    log_message("Spreading activation activated " +
                std::to_string(spread_stats.concepts_activated) + " concepts");

    // PHASE 2: Compute salience to identify important concepts
    std::vector<ConceptId> all_concepts;
    for (uint64_t i = 1; i <= spread_stats.concepts_activated; ++i) {
        // This is a simplification - in real implementation, would query STM for active concepts
        all_concepts.push_back(i);
    }

    auto salience_scores = cognitive_dynamics.compute_salience_batch(
        all_concepts, context, ltm, stm, 0
    );

    // Extract top salient concepts
    std::vector<ConceptId> salient_concepts;
    for (size_t i = 0; i < salience_scores.size() && i < 10; ++i) {
        salient_concepts.push_back(salience_scores[i].concept_id);
    }

    log_message("Identified " + std::to_string(salient_concepts.size()) + " salient concepts");

    // PHASE 3: Apply Understanding Layer to salient concepts
    result.meaning_proposals = analyze_meaning(salient_concepts, ltm, stm, context);
    result.hypothesis_proposals = propose_hypotheses(salient_concepts, ltm, stm, context);
    result.contradiction_proposals = check_contradictions(salient_concepts, ltm, stm, context);

    // PHASE 4: Find analogies (if we have enough concepts)
    if (salient_concepts.size() >= 4) {
        std::vector<ConceptId> set_a(salient_concepts.begin(), salient_concepts.begin() + salient_concepts.size() / 2);
        std::vector<ConceptId> set_b(salient_concepts.begin() + salient_concepts.size() / 2, salient_concepts.end());
        result.analogy_proposals = find_analogies(set_a, set_b, ltm, stm, context);
    }

    // Compute statistics
    result.total_proposals_generated =
        result.meaning_proposals.size() +
        result.hypothesis_proposals.size() +
        result.analogy_proposals.size() +
        result.contradiction_proposals.size();

    stats_.total_cycles_performed++;

    log_message("Understanding cycle complete: " +
                std::to_string(result.total_proposals_generated) + " total proposals");

    return result;
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

template<typename ProposalType>
std::vector<ProposalType> UnderstandingLayer::filter_proposals_by_confidence(
    const std::vector<ProposalType>& proposals,
    double min_confidence
) const {
    std::vector<ProposalType> filtered;

    for (const auto& proposal : proposals) {
        if (proposal.model_confidence >= min_confidence) {
            filtered.push_back(proposal);
        }
    }

    return filtered;
}

void UnderstandingLayer::log_message(const std::string& message) const {
    if (config_.verbose_logging) {
        std::cout << "[UnderstandingLayer] " << message << std::endl;
    }
}

// Explicit template instantiations
template std::vector<MeaningProposal> UnderstandingLayer::filter_proposals_by_confidence(
    const std::vector<MeaningProposal>&, double) const;
template std::vector<HypothesisProposal> UnderstandingLayer::filter_proposals_by_confidence(
    const std::vector<HypothesisProposal>&, double) const;
template std::vector<AnalogyProposal> UnderstandingLayer::filter_proposals_by_confidence(
    const std::vector<AnalogyProposal>&, double) const;

} // namespace brain19
