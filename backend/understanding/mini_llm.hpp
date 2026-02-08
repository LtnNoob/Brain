#pragma once

#include "understanding_proposals.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/stm.hpp"
#include <vector>
#include <string>
#include <memory>

namespace brain19 {

// =============================================================================
// MINI-LLM INTERFACE
// =============================================================================
//
// MiniLLM: Abstrakte Schnittstelle für semantische Modelle
//
// ARCHITEKTUR-VERTRAG:
// ✅ Darf Texte interpretieren
// ✅ Darf Muster erkennen
// ✅ Darf Vorschläge generieren
// ✅ Darf parallel rechnen
//
// ❌ DARF NICHT:
// - Knowledge Graph modifizieren
// - Trust-Werte setzen
// - Epistemische Entscheidungen treffen
// - FACT-Promotion durchführen
// - In LTM schreiben
// - Regelgenerierung
//
// ENFORCEMENT:
// - Alle Outputs sind HYPOTHESIS
// - READ-ONLY Zugriff auf LTM
// - Kein Zugriff auf Epistemic Core
// - Vollständiges Logging
//
class MiniLLM {
public:
    virtual ~MiniLLM() = default;

    // Model identifier
    virtual std::string get_model_id() const = 0;

    // =========================================================================
    // MEANING EXTRACTION (READ-ONLY)
    // =========================================================================

    // Extract semantic meaning from activated concepts
    // INPUT: Active concepts from STM (READ-ONLY)
    // OUTPUT: Meaning proposals (HYPOTHESIS only)
    virtual std::vector<MeaningProposal> extract_meaning(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    ) const = 0;

    // =========================================================================
    // HYPOTHESIS GENERATION (PROPOSALS ONLY)
    // =========================================================================

    // Generate hypothesis proposals based on patterns
    // INPUT: Evidence concepts (READ-ONLY)
    // OUTPUT: Hypothesis proposals (NOT accepted hypotheses!)
    virtual std::vector<HypothesisProposal> generate_hypotheses(
        const std::vector<ConceptId>& evidence_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    ) const = 0;

    // =========================================================================
    // ANALOGY DETECTION (STRUCTURAL ONLY)
    // =========================================================================

    // Detect structural analogies between concept groups
    // INPUT: Concept sets (READ-ONLY)
    // OUTPUT: Analogy proposals (structural similarity, not semantic truth)
    virtual std::vector<AnalogyProposal> detect_analogies(
        const std::vector<ConceptId>& concept_set_a,
        const std::vector<ConceptId>& concept_set_b,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    ) const = 0;

    // =========================================================================
    // CONTRADICTION DETECTION (MARKING ONLY)
    // =========================================================================

    // Detect potential contradictions (does NOT resolve them)
    // INPUT: Active concepts (READ-ONLY)
    // OUTPUT: Contradiction proposals (NOT truth judgments!)
    virtual std::vector<ContradictionProposal> detect_contradictions(
        const std::vector<ConceptId>& active_concepts,
        const LongTermMemory& ltm,  // READ-ONLY!
        const ShortTermMemory& stm,  // READ-ONLY!
        ContextId context
    ) const = 0;
};

// =============================================================================
// STUB MINI-LLM (for testing without actual LLM)
// =============================================================================
//
// StubMiniLLM: Placeholder implementation for testing
//
// Returns dummy proposals to verify:
// - Epistemische Invarianten bleiben erhalten
// - Kein LTM-Zugriff erfolgt
// - Alle Outputs HYPOTHESIS sind
//
class StubMiniLLM : public MiniLLM {
public:
    StubMiniLLM() : proposal_counter_(0) {}

    std::string get_model_id() const override {
        return "stub-mini-llm-v1.0";
    }

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
    mutable uint64_t proposal_counter_;
};

} // namespace brain19
