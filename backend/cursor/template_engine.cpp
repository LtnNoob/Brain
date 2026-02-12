#include "template_engine.hpp"
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
    switch (type) {
        case RelationType::IS_A:            return "ist ein(e)";
        case RelationType::HAS_PROPERTY:    return "hat die Eigenschaft";
        case RelationType::CAUSES:          return "verursacht";
        case RelationType::ENABLES:         return "ermoeglicht";
        case RelationType::PART_OF:         return "ist Teil von";
        case RelationType::SIMILAR_TO:      return "ist aehnlich wie";
        case RelationType::CONTRADICTS:     return "widerspricht";
        case RelationType::SUPPORTS:        return "unterstuetzt";
        case RelationType::TEMPORAL_BEFORE: return "geschieht vor";
        case RelationType::CUSTOM:          return "steht in Beziehung zu";
    }
    return "steht in Beziehung zu";
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

    for (RelationType r : relations) {
        switch (r) {
            case RelationType::CAUSES:
            case RelationType::ENABLES:
            case RelationType::TEMPORAL_BEFORE:
                ++causal_count;
                break;
            case RelationType::IS_A:
            case RelationType::HAS_PROPERTY:
            case RelationType::PART_OF:
                ++def_count;
                break;
            case RelationType::SIMILAR_TO:
            case RelationType::CONTRADICTS:
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
    if (causal_count >= def_count) {
        return TemplateType::KAUSAL_ERKLAEREND;
    }
    return TemplateType::DEFINITIONAL;
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

} // namespace brain19
