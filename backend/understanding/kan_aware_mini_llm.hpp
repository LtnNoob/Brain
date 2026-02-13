#pragma once

#include "mini_llm.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../hybrid/kan_validator.hpp"
#include "../hybrid/investigation_request.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace brain19 {

// =============================================================================
// KAN-AWARE MINI-LLM (Topology B)
// =============================================================================
//
// Replaces StubMiniLLM as the production MiniLLM.
// Reads KG (READ-ONLY) + uses KAN (MicroModel) predictions as context.
//
// ARCHITECTURE CONTRACT:
// - READ-ONLY access to LTM, Registry, Embeddings
// - All output = HYPOTHESIS proposals
// - Trust ceiling: LLM-only <= 0.3, KAN-validated <= 0.7
// - Learns from KAN feedback via pattern weight adjustment
// - Never writes to KG directly
//
class KanAwareMiniLLM : public MiniLLM {
public:
    KanAwareMiniLLM(const MicroModelRegistry& registry,
                    const EmbeddingManager& embeddings);

    // MiniLLM interface
    std::string get_model_id() const override;

    std::vector<MeaningProposal> extract_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context
    ) const override;

    std::vector<HypothesisProposal> generate_hypotheses(
        const std::vector<ConceptId>& evidence_concepts,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context
    ) const override;

    std::vector<AnalogyProposal> detect_analogies(
        const std::vector<ConceptId>& concept_set_a,
        const std::vector<ConceptId>& concept_set_b,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context
    ) const override;

    std::vector<ContradictionProposal> detect_contradictions(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context
    ) const override;

    // KAN feedback for training (non-const — mutates pattern weights)
    void train_from_validation(const std::vector<ValidationResult>& results);

    // Topology A: Investigate KAN-detected anomalies
    // Generates hypotheses from InvestigationRequests produced by KanGraphMonitor
    std::vector<HypothesisProposal> investigate_anomalies(
        const std::vector<InvestigationRequest>& requests,
        const LongTermMemory& ltm,
        const ShortTermMemory& stm,
        ContextId context) const;

    // Trust ceiling constants (Topology B)
    static constexpr double LLM_ONLY_TRUST_CEILING = 0.3;
    static constexpr double KAN_VALIDATED_TRUST_CEILING = 0.7;

private:
    const MicroModelRegistry& registry_;  // READ-ONLY
    const EmbeddingManager& embeddings_;  // READ-ONLY

    // Learned pattern priorities (adjusted by KAN feedback)
    struct PatternWeights {
        double shared_parent = 1.0;        // IS_A generalization
        double transitive_causation = 1.0; // A->B->C transitive chains
        double missing_link = 1.0;         // KAN predicts high, no relation exists
        double weak_strengthening = 1.0;   // Low LTM weight, high KAN score
        double contradictory_signal = 1.0; // KAN vs LTM weight mismatch
    };
    PatternWeights weights_;
    mutable uint64_t proposal_counter_ = 0;

    // Core helper: use MicroModel to predict edge strength
    double predict_edge(ConceptId from, ConceptId to, RelationType type) const;
};

} // namespace brain19
