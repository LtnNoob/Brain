#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

namespace brain19 {

// Source type for knowledge proposals
enum class SourceType {
    WIKIPEDIA,
    GOOGLE_SCHOLAR,
    UNKNOWN
};

// Suggested epistemic type (NOT authoritative)
enum class SuggestedEpistemicType {
    FACT_CANDIDATE,
    THEORY_CANDIDATE,
    HYPOTHESIS_CANDIDATE,
    DEFINITION_CANDIDATE,
    UNKNOWN_CANDIDATE
};

// Suggested concept extracted from source
struct SuggestedConcept {
    std::string label;
    std::string context_snippet;  // Where it appeared
    
    SuggestedConcept() = default;
    SuggestedConcept(const std::string& lbl, const std::string& ctx)
        : label(lbl), context_snippet(ctx) {}
};

// Suggested relation between concepts
struct SuggestedRelation {
    std::string source_label;
    std::string target_label;
    std::string relation_type;  // "is-a", "part-of", etc.
    std::string evidence_text;
    
    SuggestedRelation() = default;
    SuggestedRelation(const std::string& src, const std::string& tgt,
                     const std::string& type, const std::string& evidence)
        : source_label(src), target_label(tgt), 
          relation_type(type), evidence_text(evidence) {}
};

// KnowledgeProposal: Pure data structure for PROPOSALS ONLY
// 
// CRITICAL EPISTEMIC RULE:
// - Importers MUST NOT assign EpistemicType
// - Importers MUST NOT assign Trust
// - Importers MUST NOT assign EpistemicStatus
// - Only SUGGESTIONS are allowed (SuggestedEpistemicType)
// - Human MUST explicitly decide epistemic metadata before LTM ingestion
//
// Requires explicit human confirmation before entering Brain19 LTM
struct KnowledgeProposal {
    // Identification
    uint64_t proposal_id;
    
    // Source metadata
    SourceType source_type;
    std::string source_reference;  // URL or DOI
    std::chrono::system_clock::time_point import_timestamp;
    
    // Extracted content
    std::string extracted_text;
    std::string title;
    
    // Suggestions for human review (NOT authoritative)
    std::vector<SuggestedConcept> suggested_concepts;
    std::vector<SuggestedRelation> suggested_relations;
    SuggestedEpistemicType suggested_epistemic_type;  // SUGGESTION ONLY
    
    // Human review guidance
    std::string notes_for_human_review;
    
    // Additional metadata (optional)
    std::vector<std::string> authors;
    std::string publication_venue;
    std::string publication_year;
    bool is_preprint;
    
    // EPISTEMIC ENFORCEMENT:
    // NO epistemic metadata fields here
    // Epistemic decisions happen ONLY during human-approved LTM ingestion
    // This struct is for PROPOSALS, not accepted knowledge
    
    // Default constructor
    KnowledgeProposal()
        : proposal_id(0)
        , source_type(SourceType::UNKNOWN)
        , import_timestamp(std::chrono::system_clock::now())
        , suggested_epistemic_type(SuggestedEpistemicType::UNKNOWN_CANDIDATE)
        , is_preprint(false)
    {}
};

} // namespace brain19
