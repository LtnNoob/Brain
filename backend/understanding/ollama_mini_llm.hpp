#pragma once

#include "mini_llm.hpp"
#include "../llm/ollama_client.hpp"
#include <sstream>

namespace brain19 {

// =============================================================================
// OLLAMA MINI-LLM
// =============================================================================
//
// OllamaMiniLLM: Real semantic analysis using Ollama
//
// ARCHITECTURE:
// - Uses existing OllamaClient for API calls
// - Generates prompts from activated concepts (READ-ONLY)
// - Parses LLM responses into proposals
// - ALL outputs are HYPOTHESIS (enforced)
//
// EPISTEMIC ENFORCEMENT:
// - READ-ONLY LTM access
// - All proposals are HYPOTHESIS
// - No trust modification
// - No knowledge writes
//
class OllamaMiniLLM : public MiniLLM {
public:
    explicit OllamaMiniLLM(const OllamaConfig& config = OllamaConfig());
    ~OllamaMiniLLM() override;

    std::string get_model_id() const override;

    // Check if Ollama is available
    bool is_available() const;

    // Semantic analysis methods (implementing MiniLLM interface)
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

private:
    OllamaClient ollama_;
    OllamaConfig config_;
    mutable uint64_t proposal_counter_;

    // Helper: Build prompt from concepts (READ-ONLY LTM access)
    std::string build_concept_description(
        const std::vector<ConceptId>& concepts,
        const LongTermMemory& ltm
    ) const;

    // Helper: Parse LLM response into structured proposal
    std::string extract_interpretation(const std::string& response) const;
    std::string extract_reasoning(const std::string& response) const;
    double estimate_confidence(const std::string& response) const;
};

} // namespace brain19
