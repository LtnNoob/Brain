#pragma once

#include "traversal_types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/active_relation.hpp"
#include <string>
#include <vector>

namespace brain19 {

// =============================================================================
// TEMPLATE ENGINE
// =============================================================================
//
// Day-1 language output: converts FocusCursor traversal chains into
// German natural language using relation-type → sentence-pattern mapping.
//
// No LLM needed. Templates come directly from the FocusCursor chain.
//
// Example:
//   Chain: Eis →CAUSES→ Schmelzen →CAUSES→ Wasser →HAS_PROPERTY→ Fluessig
//   Output: "Eis verursacht Schmelzen. Schmelzen verursacht Wasser.
//            Wasser hat die Eigenschaft Fluessig."
//

// Template types for different chain structures
enum class TemplateType {
    KAUSAL_ERKLAEREND,   // Chain dominated by CAUSES/ENABLES
    DEFINITIONAL,        // Chain dominated by IS_A/HAS_PROPERTY
    AUFZAEHLEND,         // Branching / listing pattern
    VERGLEICHEND         // Contains SIMILAR_TO / CONTRADICTS
};

// Result of template generation
struct TemplateResult {
    std::string text;                   // Generated German text
    TemplateType template_type;         // Detected template type
    size_t sentences_generated = 0;     // Number of sentences
    double confidence = 1.0;            // Always 1.0 for templates
};

class TemplateEngine {
public:
    explicit TemplateEngine(const LongTermMemory& ltm);

    // Generate text from a traversal chain
    TemplateResult generate(const TraversalResult& chain) const;

    // Generate text from concept/relation sequences directly
    TemplateResult generate(const std::vector<ConceptId>& concepts,
                            const std::vector<RelationType>& relations) const;

    // Classify chain into template type
    TemplateType classify(const std::vector<RelationType>& relations) const;

    // Get sentence pattern for a single relation step
    std::string relation_sentence(const std::string& source_label,
                                   const std::string& target_label,
                                   RelationType type) const;

    // Get relation type name in German
    static std::string relation_name_de(RelationType type);

private:
    const LongTermMemory& ltm_;

    // Look up concept label, fallback to ID string
    std::string concept_label(ConceptId id) const;
};

} // namespace brain19
