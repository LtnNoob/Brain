#include "template_engine.hpp"
#include "../memory/relation_type_registry.hpp"
#include <sstream>
#include <algorithm>

namespace brain19 {

TemplateEngine::TemplateEngine(const LongTermMemory& ltm)
    : ltm_(ltm)
{
}

std::string TemplateEngine::concept_label(ConceptId id) const {
    auto info = ltm_.retrieve_concept(id);
    if (info) return info->label;
    return "Konzept#" + std::to_string(id);
}

std::string TemplateEngine::relation_name_de(RelationType type) {
    return RelationTypeRegistry::instance().get_name_de(type);
}

std::string TemplateEngine::relation_sentence(
    const std::string& source_label,
    const std::string& target_label,
    RelationType type
) const {
    return source_label + " " + relation_name_de(type) + " " + target_label + ".";
}

TemplateType TemplateEngine::classify(const std::vector<RelationType>& relations) const {
    if (relations.empty()) return TemplateType::DEFINITIONAL;

    size_t causal_count = 0;
    size_t def_count = 0;
    size_t compare_count = 0;

    auto& reg = RelationTypeRegistry::instance();
    for (RelationType r : relations) {
        auto cat = reg.get_category(r);
        switch (cat) {
            case RelationCategory::CAUSAL:
            case RelationCategory::TEMPORAL:
                ++causal_count;
                break;
            case RelationCategory::HIERARCHICAL:
            case RelationCategory::COMPOSITIONAL:
                ++def_count;
                break;
            case RelationCategory::SIMILARITY:
            case RelationCategory::OPPOSITION:
                ++compare_count;
                break;
            default:
                break;
        }
    }

    // Classify by dominant relation category
    if (compare_count > 0 && compare_count >= causal_count && compare_count >= def_count) {
        return TemplateType::VERGLEICHEND;
    }
    if (causal_count > def_count) {
        return TemplateType::KAUSAL_ERKLAEREND;
    }
    if (def_count > causal_count) {
        return TemplateType::DEFINITIONAL;
    }
    // Tie or all-zero: default to AUFZAEHLEND (mixed/listing pattern)
    if (causal_count == 0 && def_count == 0) {
        return TemplateType::AUFZAEHLEND;
    }
    return TemplateType::KAUSAL_ERKLAEREND;  // True tie: causal wins
}

TemplateResult TemplateEngine::generate(const TraversalResult& chain) const {
    return generate(chain.concept_sequence, chain.relation_sequence);
}

TemplateResult TemplateEngine::generate(
    const std::vector<ConceptId>& concepts,
    const std::vector<RelationType>& relations
) const {
    TemplateResult result;
    result.template_type = classify(relations);
    result.sentences_generated = 0;

    if (concepts.empty()) {
        result.text = "";
        return result;
    }

    // Single concept, no relations
    if (concepts.size() == 1) {
        auto info = ltm_.retrieve_concept(concepts[0]);
        if (info && !info->definition.empty()) {
            result.text = info->label + ": " + info->definition + ".";
        } else {
            result.text = concept_label(concepts[0]) + ".";
        }
        result.sentences_generated = 1;
        return result;
    }

    // Build sentences from chain
    std::ostringstream oss;
    size_t num_edges = std::min(relations.size(), concepts.size() - 1);

    for (size_t i = 0; i < num_edges; ++i) {
        if (i > 0) oss << " ";
        oss << relation_sentence(
            concept_label(concepts[i]),
            concept_label(concepts[i + 1]),
            relations[i]
        );
        ++result.sentences_generated;
    }

    result.text = oss.str();
    return result;
}

// =============================================================================
// English templates (Convergence v2, Section 13)
// =============================================================================

std::string TemplateEngine::relation_template_en(RelationType type) {
    switch (type) {
        case RelationType::IS_A:              return "{subject} is a {object}";
        case RelationType::INSTANCE_OF:       return "{subject} is an instance of {object}";
        case RelationType::DERIVED_FROM:      return "{subject} is derived from {object}";
        case RelationType::HAS_PROPERTY:      return "{subject} is {object}";
        case RelationType::PART_OF:           return "{subject} is part of {object}";
        case RelationType::HAS_PART:          return "{subject} has {object}";
        case RelationType::CAUSES:            return "{subject} causes {object}";
        case RelationType::ENABLES:           return "{subject} enables {object}";
        case RelationType::PRODUCES:          return "{subject} produces {object}";
        case RelationType::IMPLIES:           return "{subject} implies {object}";
        case RelationType::CONTRADICTS:       return "{subject} contradicts {object}";
        case RelationType::SUPPORTS:          return "{subject} supports {object}";
        case RelationType::SIMILAR_TO:        return "{subject} is similar to {object}";
        case RelationType::ASSOCIATED_WITH:   return "{subject} is associated with {object}";
        case RelationType::TEMPORAL_BEFORE:   return "{subject} occurs before {object}";
        case RelationType::TEMPORAL_AFTER:    return "{subject} occurs after {object}";
        case RelationType::REQUIRES:          return "{subject} requires {object}";
        case RelationType::USES:              return "{subject} uses {object}";
        case RelationType::SOURCE:            return "{subject} originates from {object}";
        default:                              return "{subject} is related to {object}";
    }
}

std::string TemplateEngine::relation_sentence_en(
    const std::string& subject,
    const std::string& object,
    RelationType type
) const {
    std::string tmpl = relation_template_en(type);
    // Replace {subject} and {object} placeholders
    std::string result = tmpl;
    auto pos = result.find("{subject}");
    if (pos != std::string::npos) result.replace(pos, 9, subject);
    pos = result.find("{object}");
    if (pos != std::string::npos) result.replace(pos, 8, object);
    return result + ".";
}

TemplateResult TemplateEngine::generate_en(const TraversalResult& chain) const {
    return generate_en(chain.concept_sequence, chain.relation_sequence);
}

TemplateResult TemplateEngine::generate_en(
    const std::vector<ConceptId>& concepts,
    const std::vector<RelationType>& relations
) const {
    TemplateResult result;
    result.template_type = classify(relations);
    result.sentences_generated = 0;

    if (concepts.empty()) {
        result.text = "";
        return result;
    }

    if (concepts.size() == 1) {
        auto info = ltm_.retrieve_concept(concepts[0]);
        if (info && !info->definition.empty()) {
            result.text = info->label + ": " + info->definition + ".";
        } else {
            result.text = concept_label(concepts[0]) + ".";
        }
        result.sentences_generated = 1;
        return result;
    }

    std::ostringstream oss;
    size_t num_edges = std::min(relations.size(), concepts.size() - 1);

    for (size_t i = 0; i < num_edges; ++i) {
        if (i > 0) oss << " ";
        oss << relation_sentence_en(
            concept_label(concepts[i]),
            concept_label(concepts[i + 1]),
            relations[i]
        );
        ++result.sentences_generated;
    }

    result.text = oss.str();
    return result;
}

// =============================================================================
// Epistemic modality framing (Convergence v2, Section 13)
// =============================================================================

std::string TemplateEngine::epistemic_frame(
    float trust, EpistemicType type, const std::string& sentence
) {
    // Epistemic type overrides trust for low-confidence categories
    if (type == EpistemicType::SPECULATION) {
        return "it's speculated that " + sentence;
    }
    if (type == EpistemicType::HYPOTHESIS) {
        return "it's hypothesized that " + sentence;
    }

    // Trust-based modality for FACT/THEORY/DEFINITION/INFERENCE
    if (trust >= 0.95f) return sentence;
    if (trust >= 0.85f) return "generally, " + sentence;
    if (trust >= 0.60f) return "likely, " + sentence;

    // Low trust: hedge with "might be"
    if (trust >= 0.30f) {
        std::string hedged = sentence;
        auto pos = hedged.find(" is ");
        if (pos != std::string::npos) {
            hedged.replace(pos, 4, " might be ");
        } else {
            hedged = "possibly, " + hedged;
        }
        return hedged;
    }

    return "it's speculated that " + sentence;
}

} // namespace brain19
