#pragma once

#include "concept_model_registry.hpp"
#include "../understanding/mini_llm.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../hybrid/kan_validator.hpp"
#include "../hybrid/investigation_request.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace brain19 {

// =============================================================================
// CONCEPT PATTERN ENGINE
// =============================================================================
//
// Replaces KanAwareMiniLLM. Implements MiniLLM interface using per-concept
// ConceptModel predictions and per-concept pattern weights.
//
// Key difference from KanAwareMiniLLM: pattern weights are PER-CONCEPT,
// not global. Concept "Dog" and concept "Algebra" develop different
// pattern weight profiles based on validation feedback.
//
class ConceptPatternEngine : public MiniLLM {
public:
    ConceptPatternEngine(const ConceptModelRegistry& registry,
                         const EmbeddingManager& embeddings);

    std::string get_model_id() const override;

    std::vector<MeaningProposal> extract_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context) const override;

    std::vector<HypothesisProposal> generate_hypotheses(
        const std::vector<ConceptId>& evidence_concepts,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context,
        const std::vector<ThoughtPath>& thought_paths = {}) const override;

    std::vector<AnalogyProposal> detect_analogies(
        const std::vector<ConceptId>& concept_set_a,
        const std::vector<ConceptId>& concept_set_b,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context) const override;

    std::vector<ContradictionProposal> detect_contradictions(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context) const override;

    // KAN feedback (non-const — mutates per-concept pattern weights)
    void train_from_validation(const std::vector<ValidationResult>& results);

    // Topology A investigation
    std::vector<HypothesisProposal> investigate_anomalies(
        const std::vector<InvestigationRequest>& requests,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context) const;

    static constexpr double LLM_ONLY_TRUST_CEILING = 0.3;
    static constexpr double KAN_VALIDATED_TRUST_CEILING = 0.7;

private:
    const ConceptModelRegistry& registry_;
    const EmbeddingManager& embeddings_;
    mutable uint64_t proposal_counter_ = 0;

    // Helper: predict edge via ConceptModel
    double predict_edge(ConceptId from, ConceptId to, RelationType type) const;
    // Helper: get pattern weight for a concept (falls back to defaults)
    double get_pattern_weight(ConceptId cid, size_t pattern_idx) const;
};

} // namespace brain19
