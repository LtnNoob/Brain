#pragma once

#include "../ltm/long_term_memory.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include "foundation_concepts.hpp"
#include "context_accumulator.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace brain19 {

// BootstrapProposal: A candidate concept extracted from text for human review
struct BootstrapProposal {
    std::string entity_name;
    std::string context_text;                   // Original text where found
    std::vector<std::string> suggested_types;   // Based on existing knowledge
    std::vector<std::string> similar_concepts;  // From LTM
    double suggested_trust;
    std::string auto_description;               // Generated from context
};

// BootstrapInterface: Guided onboarding for Brain19
//
// Solves the bootstrap problem:
// 1. Seeds foundation knowledge (ontology, categories, relations, science)
// 2. Processes new text → extracts entity candidates
// 3. Presents proposals to human for guided review
// 4. Accumulates context for progressively better suggestions
class BootstrapInterface {
public:
    explicit BootstrapInterface(LongTermMemory& ltm);

    // ─── Foundation ────────────────────────────────────────────────────
    void initialize_foundation();
    bool is_foundation_loaded() const;

    // ─── Text Processing ───────────────────────────────────────────────
    // Extract candidate entities from text and generate proposals
    std::vector<BootstrapProposal> process_text(const std::string& text);

    // ─── Human Review ──────────────────────────────────────────────────
    void accept_proposal(const BootstrapProposal& p,
                         const std::string& human_description,
                         EpistemicType type, double trust);

    void reject_proposal(const BootstrapProposal& p,
                         const std::string& reason);

    // ─── Stats ─────────────────────────────────────────────────────────
    size_t known_concepts() const;
    size_t pending_proposals() const;

    // ─── Progressive Complexity ────────────────────────────────────────
    // Suggest topics the human should teach next (based on gaps)
    std::vector<std::string> suggest_next_topics() const;

    // Access accumulator for detailed metrics
    const ContextAccumulator& accumulator() const { return accumulator_; }

private:
    LongTermMemory& ltm_;
    ContextAccumulator accumulator_;
    bool foundation_loaded_;

    // Label index for fast duplicate / similarity checks
    std::unordered_map<std::string, ConceptId> label_index_;
    // Pending proposals not yet accepted/rejected
    std::vector<BootstrapProposal> pending_;
    // Rejected entity names (avoid re-proposing)
    std::unordered_set<std::string> rejected_;

    // Internal helpers
    void rebuild_label_index();
    std::vector<std::string> find_similar(const std::string& name) const;
    std::vector<std::string> extract_candidate_entities(const std::string& text) const;
    std::string generate_description(const std::string& entity,
                                      const std::string& context) const;
    bool is_known(const std::string& label) const;
};

} // namespace brain19
