#pragma once

#include "mini_llm.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/active_relation.hpp"      // RelationType
#include "../epistemic/epistemic_metadata.hpp" // EpistemicType
#include <memory>
#include <string>
#include <vector>

namespace brain19 {

// =============================================================================
// MINI-LLM FACTORY
// =============================================================================
//
// MiniLLMFactory: Erzeugt spezialisierte Mini-LLMs für gelernte Konzepte
//
// KONZEPT:
// - Wenn Brain19 etwas Neues lernt, wird ein Mini-LLM dafür erzeugt
// - Jedes Mini-LLM ist Experte für einen bestimmten Wissensbereich
// - Mini-LLMs können parallel arbeiten
// - Understanding Layer sammelt ihre Vorschläge
//
// ARCHITEKTUR:
// - Factory erzeugt Mini-LLMs basierend auf LTM-Konzepten
// - Jedes Mini-LLM bekommt einen Kontext (fine-tuning via prompt)
// - Mini-LLMs bleiben READ-ONLY bzgl. LTM
// - Alle Outputs sind HYPOTHESIS
//
// BEISPIEL:
//   Brain19 lernt "Katzen sind Säugetiere"
//   → Factory erzeugt "cat-mammal-mini-llm"
//   → Dieses LLM ist Experte für Katzen/Säugetiere
//   → Bei Fragen zu Katzen wird dieses LLM bevorzugt
//
// TODO: Not yet implemented — planned for KAN-LLM Hybrid Layer
class MiniLLMFactory {
public:
    MiniLLMFactory();
    ~MiniLLMFactory() = default;

    // =========================================================================
    // DYNAMIC MINI-LLM CREATION
    // =========================================================================

    // Create specialized Mini-LLM from concept(s)
    // When Brain19 learns something, this is called to create an expert
    std::unique_ptr<MiniLLM> create_specialized_mini_llm(
        const std::vector<ConceptId>& focal_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const std::string& specialization_name
    );

    // Create Mini-LLM from relation pattern
    // E.g., "All IS_A relations between mammals"
    std::unique_ptr<MiniLLM> create_relation_expert_mini_llm(
        RelationType relation_type,
        const std::vector<ConceptId>& domain_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const std::string& specialization_name
    );

    // Create Mini-LLM from epistemic type
    // E.g., "Expert for all THEORIES in physics"
    std::unique_ptr<MiniLLM> create_epistemic_expert_mini_llm(
        EpistemicType type,
        const std::vector<ConceptId>& domain_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const std::string& specialization_name
    );

    // =========================================================================
    // MINI-LLM MANAGEMENT
    // =========================================================================

    // Get count of created Mini-LLMs
    size_t get_created_count() const { return created_count_; }

private:
    // Placeholder for future KAN-based factory config
    size_t created_count_ = 0;

    // Helper: Build specialized system prompt for Mini-LLM
    std::string build_specialization_prompt(
        const std::vector<ConceptId>& concepts,
        const LongTermMemory& ltm
    ) const;

    // Helper: Extract knowledge context (READ-ONLY)
    std::string extract_knowledge_context(
        const std::vector<ConceptId>& concepts,
        const LongTermMemory& ltm
    ) const;
};

// =============================================================================
// SPECIALIZED MINI-LLM
// =============================================================================
//
// SpecializedMiniLLM: Ein Mini-LLM mit Expertise in einem spezifischen Bereich
//
// EIGENSCHAFTEN:
// - Hat einen "Kontext" (Wissen über spezifische Konzepte)
// - System-Prompt enthält dieses Wissen
// - Wird bevorzugt für Fragen in seinem Bereich
// - Bleibt READ-ONLY bzgl. LTM
//
// TODO: Not yet implemented — planned for KAN-LLM Hybrid Layer
class SpecializedMiniLLM : public MiniLLM {
public:
    SpecializedMiniLLM(
        const std::string& name,
        const std::string& specialization_context
    );

    ~SpecializedMiniLLM() override = default;

    std::string get_model_id() const override;

    // Get specialization info
    const std::string& get_specialization_context() const {
        return specialization_context_;
    }

    // Check if this Mini-LLM is relevant for given concepts
    bool is_relevant_for(
        const std::vector<ConceptId>& concepts,
        const LongTermMemory& ltm
    ) const;

    // Implementing MiniLLM interface
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
    std::string name_;
    std::string specialization_context_;  // Knowledge this Mini-LLM specializes in
    std::vector<ConceptId> focal_concepts_;  // Concepts this Mini-LLM knows about
    mutable uint64_t proposal_counter_ = 0;
};

} // namespace brain19
