#pragma once

#include "../epistemic/epistemic_metadata.hpp"
#include "../ltm/long_term_memory.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace brain19 {

// =============================================================================
// MEANING PROPOSAL
// =============================================================================
//
// MeaningProposal: Semantischer Vorschlag eines Mini-LLMs
//
// KRITISCH:
// - Dies ist KEIN Wissen
// - Dies ist KEINE Wahrheit
// - Dies ist NUR ein Vorschlag
// - Epistemic Core entscheidet über Akzeptanz
//
struct MeaningProposal {
    uint64_t proposal_id;

    // Source concepts (READ-ONLY references)
    std::vector<ConceptId> source_concepts;

    // Proposed interpretation
    std::string interpretation;
    std::string reasoning;

    // Confidence (NOT trust! This is model confidence, not epistemic trust)
    double model_confidence;  // [0.0, 1.0]

    // Mini-LLM identifier
    std::string source_model;

    // EPISTEMIC STATUS:
    // ALL proposals are HYPOTHESIS until validated by Epistemic Core
    // This field is ALWAYS HYPOTHESIS and CANNOT be changed here
    EpistemicType epistemic_type = EpistemicType::HYPOTHESIS;

    MeaningProposal(
        uint64_t id,
        const std::vector<ConceptId>& sources,
        const std::string& interp,
        const std::string& reason,
        double confidence,
        const std::string& model
    ) : proposal_id(id)
      , source_concepts(sources)
      , interpretation(interp)
      , reasoning(reason)
      , model_confidence(std::max(0.0, std::min(1.0, confidence)))
      , source_model(model)
      , epistemic_type(EpistemicType::HYPOTHESIS)  // ALWAYS HYPOTHESIS
    {
    }
};

// =============================================================================
// HYPOTHESIS PROPOSAL
// =============================================================================
//
// HypothesisProposal: Vorgeschlagene Hypothese basierend auf Muster-Erkennung
//
// KRITISCH:
// - Dies ist eine VORGESCHLAGENE Hypothese
// - NICHT eine akzeptierte Hypothese
// - Epistemic Core entscheidet über Aufnahme in LTM
// - Understanding Layer darf NICHT in LTM schreiben
//
struct HypothesisProposal {
    uint64_t proposal_id;

    // Source evidence (READ-ONLY references)
    std::vector<ConceptId> evidence_concepts;

    // Proposed hypothesis content
    std::string hypothesis_statement;
    std::string supporting_reasoning;

    // Detected patterns
    std::vector<std::string> detected_patterns;

    // Model confidence (NOT epistemic trust!)
    double model_confidence;  // [0.0, 1.0]

    // Mini-LLM identifier
    std::string source_model;

    // Suggested epistemic metadata (SUGGESTION ONLY!)
    // Epistemic Core MUST validate and decide
    struct SuggestedEpistemic {
        EpistemicType suggested_type = EpistemicType::HYPOTHESIS;  // Always HYPOTHESIS
        double suggested_trust = 0.5;  // Conservative default

        SuggestedEpistemic() = default;

        SuggestedEpistemic(EpistemicType type, double trust)
            : suggested_type(EpistemicType::HYPOTHESIS)  // ALWAYS HYPOTHESIS
            , suggested_trust(std::max(0.0, std::min(1.0, trust)))
        {
            // ENFORCEMENT: Type is ALWAYS HYPOTHESIS regardless of input
        }
    };

    SuggestedEpistemic suggested_epistemic;

    HypothesisProposal(
        uint64_t id,
        const std::vector<ConceptId>& evidence,
        const std::string& statement,
        const std::string& reasoning,
        const std::vector<std::string>& patterns,
        double confidence,
        const std::string& model
    ) : proposal_id(id)
      , evidence_concepts(evidence)
      , hypothesis_statement(statement)
      , supporting_reasoning(reasoning)
      , detected_patterns(patterns)
      , model_confidence(std::max(0.0, std::min(1.0, confidence)))
      , source_model(model)
      , suggested_epistemic(EpistemicType::HYPOTHESIS, confidence)
    {
    }
};

// =============================================================================
// ANALOGY PROPOSAL
// =============================================================================
//
// AnalogyProposal: Vorgeschlagene strukturelle Analogie
//
// KRITISCH:
// - Analogien sind VORSCHLÄGE, keine Fakten
// - Understanding Layer erkennt nur Muster
// - Epistemic Core validiert semantische Korrektheit
//
struct AnalogyProposal {
    uint64_t proposal_id;

    // Source domain concepts
    std::vector<ConceptId> source_domain;

    // Target domain concepts
    std::vector<ConceptId> target_domain;

    // Structural mapping
    std::string mapping_description;

    // Similarity score (structural, not semantic)
    double structural_similarity;  // [0.0, 1.0]

    // Model confidence
    double model_confidence;  // [0.0, 1.0]

    // Mini-LLM identifier
    std::string source_model;

    AnalogyProposal(
        uint64_t id,
        const std::vector<ConceptId>& source,
        const std::vector<ConceptId>& target,
        const std::string& mapping,
        double similarity,
        double confidence,
        const std::string& model
    ) : proposal_id(id)
      , source_domain(source)
      , target_domain(target)
      , mapping_description(mapping)
      , structural_similarity(std::max(0.0, std::min(1.0, similarity)))
      , model_confidence(std::max(0.0, std::min(1.0, confidence)))
      , source_model(model)
    {
    }
};

// =============================================================================
// CONTRADICTION PROPOSAL
// =============================================================================
//
// ContradictionProposal: Erkannte potenzielle Inkonsistenz
//
// KRITISCH:
// - Dies ist eine VERMUTETE Inkonsistenz
// - Epistemic Core entscheidet über tatsächliche Inkonsistenz
// - Understanding Layer darf NICHT Trust ändern
// - NUR Markierung, keine Modifikation
//
struct ContradictionProposal {
    uint64_t proposal_id;

    // Potentially contradicting concepts
    ConceptId concept_a;
    ConceptId concept_b;

    // Description of contradiction
    std::string contradiction_description;
    std::string reasoning;

    // Severity (model's assessment, not epistemic judgment)
    double severity;  // [0.0, 1.0]

    // Model confidence in this being a contradiction
    double model_confidence;  // [0.0, 1.0]

    // Mini-LLM identifier
    std::string source_model;

    ContradictionProposal(
        uint64_t id,
        ConceptId a,
        ConceptId b,
        const std::string& description,
        const std::string& reason,
        double sev,
        double confidence,
        const std::string& model
    ) : proposal_id(id)
      , concept_a(a)
      , concept_b(b)
      , contradiction_description(description)
      , reasoning(reason)
      , severity(std::max(0.0, std::min(1.0, sev)))
      , model_confidence(std::max(0.0, std::min(1.0, confidence)))
      , source_model(model)
    {
    }
};

} // namespace brain19
