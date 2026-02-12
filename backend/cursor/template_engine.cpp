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

} // namespace brain19
