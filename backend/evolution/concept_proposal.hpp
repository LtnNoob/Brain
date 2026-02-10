#pragma once

#include "../common/types.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include "../curiosity/curiosity_trigger.hpp"
#include "../understanding/understanding_proposals.hpp"
#include "../micromodel/relevance_map.hpp"
#include "../ltm/long_term_memory.hpp"
#include <string>
#include <vector>
#include <optional>

namespace brain19 {

// =============================================================================
// CONCEPT PROPOSAL
// =============================================================================
//
// System-generated proposals for NEW concepts based on:
// - CuriosityEngine triggers (knowledge gaps)
// - RelevanceMap anomalies (unexpected connections)
// - Cross-concept analogies
//
// CRITICAL INVARIANT:
// - System-generated concepts ALWAYS start as SPECULATION or HYPOTHESIS
// - Initial trust is CAPPED at 0.5
// - Only the epistemic promotion pipeline can elevate status
//

struct ConceptProposal {
    std::string label;
    std::string description;
    EpistemicType initial_type;       // Always SPECULATION or HYPOTHESIS
    double initial_trust;              // Max 0.5 for system-generated
    std::string source;                // "curiosity:shallow_relations", "analogy:X↔Y", etc.
    std::vector<ConceptId> evidence;   // Supporting concepts
    std::string reasoning;             // Why this concept is proposed

    ConceptProposal(
        const std::string& lbl,
        const std::string& desc,
        EpistemicType type,
        double trust,
        const std::string& src,
        const std::vector<ConceptId>& ev,
        const std::string& reason
    ) : label(lbl)
      , description(desc)
      , initial_type(type)
      , initial_trust(std::min(trust, 0.5))  // ENFORCE max 0.5
      , source(src)
      , evidence(ev)
      , reasoning(reason)
    {
        // Enforce: only SPECULATION or HYPOTHESIS for system-generated
        if (initial_type != EpistemicType::SPECULATION &&
            initial_type != EpistemicType::HYPOTHESIS) {
            initial_type = EpistemicType::SPECULATION;
        }
    }
};

class ConceptProposer {
public:
    explicit ConceptProposer(const LongTermMemory& ltm);

    // Generate proposals from curiosity triggers
    std::vector<ConceptProposal> from_curiosity(
        const std::vector<CuriosityTrigger>& triggers);

    // Generate proposals from relevance anomalies
    std::vector<ConceptProposal> from_relevance_anomalies(
        const RelevanceMap& map, double anomaly_threshold = 0.8);

    // Generate proposals from cross-concept analogies
    std::vector<ConceptProposal> from_analogies(
        const std::vector<AnalogyProposal>& analogies);

    // Deduplicate and rank proposals
    std::vector<ConceptProposal> rank_proposals(
        std::vector<ConceptProposal>& proposals, size_t max_k = 10);

private:
    const LongTermMemory& ltm_;

    // Check if a similar concept already exists
    bool concept_exists_similar(const std::string& label) const;

    // Compute proposal quality score for ranking
    double compute_quality_score(const ConceptProposal& proposal) const;
};

} // namespace brain19
