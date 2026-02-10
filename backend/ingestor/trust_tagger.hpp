#pragma once

#include "../epistemic/epistemic_metadata.hpp"
#include "../importers/knowledge_proposal.hpp"
#include <string>
#include <vector>

namespace brain19 {

// TrustCategory: High-level trust classification for incoming knowledge
//
// Maps to the existing EpistemicType + trust values:
//   FACTS        → EpistemicType::FACT,       trust 0.98-0.99
//   DEFINITIONS  → EpistemicType::DEFINITION,  trust 0.95-0.99
//   THEORIES     → EpistemicType::THEORY,      trust 0.90-0.95
//   HYPOTHESES   → EpistemicType::HYPOTHESIS,  trust 0.50-0.80
//   INFERENCES   → EpistemicType::INFERENCE,   trust 0.40-0.70
//   SPECULATION  → EpistemicType::SPECULATION,  trust 0.20-0.30
//   INVALIDATED  → EpistemicStatus::INVALIDATED, trust 0.05
enum class TrustCategory {
    FACTS,
    DEFINITIONS,
    THEORIES,
    HYPOTHESES,
    INFERENCES,
    SPECULATION,
    INVALIDATED
};

// TrustAssignment: The result of trust tagging
struct TrustAssignment {
    TrustCategory category;
    EpistemicType epistemic_type;
    EpistemicStatus epistemic_status;
    double trust_value;
    std::string reasoning;      // Why this trust level was assigned

    // Create the EpistemicMetadata from this assignment
    EpistemicMetadata to_epistemic_metadata() const {
        return EpistemicMetadata(epistemic_type, epistemic_status, trust_value);
    }
};

// TrustTagger: Maps trust categories to the existing epistemic system
//
// DESIGN:
// - Bridges high-level trust categories to brain19's EpistemicMetadata
// - Uses heuristic signals from text to suggest trust level
// - All assignments are SUGGESTIONS - human review decides final trust
// - Respects existing epistemic integrity constraints
//
// SIGNALS USED FOR TRUST ESTIMATION:
// - Source reliability (Wikipedia vs. random text)
// - Hedging language ("might", "possibly", "suggests")
// - Definitional patterns ("X is defined as")
// - Citation presence (suggests better-supported claims)
// - Certainty language ("proven", "established", "known")
class TrustTagger {
public:
    TrustTagger();

    // Assign trust category from explicit category
    TrustAssignment assign_trust(TrustCategory category) const;

    // Suggest trust based on source type
    TrustAssignment suggest_from_source(SourceType source) const;

    // Suggest trust based on text signals
    TrustAssignment suggest_from_text(const std::string& text) const;

    // Suggest trust from SuggestedEpistemicType (existing importer proposals)
    TrustAssignment suggest_from_proposal(SuggestedEpistemicType suggested_type) const;

    // Override trust value within a category's valid range
    TrustAssignment assign_trust_with_value(TrustCategory category, double trust) const;

    // Get the valid trust range for a category
    struct TrustRange {
        double min_trust;
        double max_trust;
        double default_trust;
    };
    TrustRange get_trust_range(TrustCategory category) const;

    // Convert category to string
    static std::string category_to_string(TrustCategory cat);

private:
    // Text analysis helpers
    bool has_hedging_language(const std::string& text) const;
    bool has_certainty_language(const std::string& text) const;
    bool has_definition_pattern(const std::string& text) const;
    bool has_citation_markers(const std::string& text) const;
    double compute_text_confidence(const std::string& text) const;
};

} // namespace brain19
