#include "relation_extractor.hpp"
#include <algorithm>
#include <regex>
#include <set>

namespace brain19 {

RelationExtractor::RelationExtractor(const Config& config)
    : config_(config)
{
    init_patterns();
}

void RelationExtractor::init_patterns() {
    // IS_A patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+is\\s+(?:a|an)\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::IS_A, 0.8, true
    });
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:are|were)\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::IS_A, 0.6, true
    });
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s*,\\s*(?:a|an)\\s+(?:type|kind|form)\\s+of\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::IS_A, 0.85, true
    });

    // HAS_PROPERTY patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+has\\s+(?:a|an)?\\s*([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::HAS_PROPERTY, 0.7, true
    });
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:contains|includes|possesses)\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::HAS_PROPERTY, 0.7, true
    });

    // CAUSES patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+causes\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::CAUSES, 0.8, true
    });
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:leads?\\s+to|results?\\s+in)\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::CAUSES, 0.7, true
    });

    // ENABLES patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:enables|allows|permits|facilitates)\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::ENABLES, 0.75, true
    });

    // PART_OF patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+is\\s+(?:a\\s+)?part\\s+of\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::PART_OF, 0.85, true
    });
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+belongs?\\s+to\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)*)",
        RelationType::PART_OF, 0.7, true
    });

    // SIMILAR_TO patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+is\\s+(?:similar|related|comparable)\\s+to\\s+([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)",
        RelationType::SIMILAR_TO, 0.7, true
    });
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+resembles?\\s+([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)",
        RelationType::SIMILAR_TO, 0.65, true
    });

    // CONTRADICTS patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:contradicts|opposes|conflicts\\s+with|refutes)\\s+([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)",
        RelationType::CONTRADICTS, 0.8, true
    });

    // SUPPORTS patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:supports|confirms|validates|corroborates)\\s+([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)",
        RelationType::SUPPORTS, 0.75, true
    });

    // TEMPORAL_BEFORE patterns
    patterns_.push_back({
        "([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:precedes|comes?\\s+before|predates)\\s+([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)",
        RelationType::TEMPORAL_BEFORE, 0.7, true
    });
}

std::vector<ExtractedRelation> RelationExtractor::extract_relations_blind(
    const std::string& text) const
{
    std::vector<ExtractedRelation> relations;

    for (const auto& pattern : patterns_) {
        try {
            std::regex regex(pattern.pattern_str);
            std::smatch match;
            std::string::const_iterator search_start = text.cbegin();

            while (std::regex_search(search_start, text.cend(), match, regex)) {
                std::string source = pattern.source_is_group1 ? match[1].str() : match[2].str();
                std::string target = pattern.source_is_group1 ? match[2].str() : match[1].str();
                std::string evidence = match[0].str();

                if (source.size() >= 2 && target.size() >= 2 &&
                    pattern.base_confidence >= config_.min_confidence) {
                    relations.emplace_back(source, target, pattern.type,
                                          evidence, pattern.base_confidence);
                }

                search_start = match.suffix().first;
                if (relations.size() >= config_.max_relations) break;
            }
        } catch (const std::regex_error&) {
            // Skip invalid patterns
            continue;
        }

        if (relations.size() >= config_.max_relations) break;
    }

    return relations;
}

std::vector<ExtractedRelation> RelationExtractor::extract_relations(
    const std::string& text,
    const std::vector<ExtractedEntity>& known_entities) const
{
    // First, do blind extraction
    auto relations = extract_relations_blind(text);

    // Then, boost confidence for relations that involve known entities
    std::set<std::string> entity_labels;
    for (const auto& entity : known_entities) {
        entity_labels.insert(entity.label);
    }

    for (auto& rel : relations) {
        bool source_known = entity_labels.count(rel.source_label) > 0;
        bool target_known = entity_labels.count(rel.target_label) > 0;

        if (source_known && target_known) {
            rel.confidence = std::min(1.0, rel.confidence * 1.3);
        } else if (source_known || target_known) {
            rel.confidence = std::min(1.0, rel.confidence * 1.1);
        }
    }

    // Also try entity-pair specific extraction for nearby entities
    for (size_t i = 0; i < known_entities.size() && relations.size() < config_.max_relations; ++i) {
        for (size_t j = i + 1; j < known_entities.size() && relations.size() < config_.max_relations; ++j) {
            auto pair_rels = extract_entity_pair_relations(
                text, known_entities[i].label, known_entities[j].label);
            relations.insert(relations.end(), pair_rels.begin(), pair_rels.end());
        }
    }

    // Deduplicate
    std::set<std::string> seen;
    std::vector<ExtractedRelation> deduped;
    for (auto& rel : relations) {
        std::string key = rel.source_label + "|" +
                          std::to_string(static_cast<int>(rel.relation_type)) + "|" +
                          rel.target_label;
        if (seen.insert(key).second) {
            deduped.push_back(std::move(rel));
        }
    }

    // Sort by confidence
    std::sort(deduped.begin(), deduped.end(),
        [](const ExtractedRelation& a, const ExtractedRelation& b) {
            return a.confidence > b.confidence;
        });

    if (deduped.size() > config_.max_relations) {
        deduped.resize(config_.max_relations);
    }

    return deduped;
}

std::vector<ExtractedRelation> RelationExtractor::extract_entity_pair_relations(
    const std::string& text,
    const std::string& entity_a,
    const std::string& entity_b) const
{
    std::vector<ExtractedRelation> relations;

    // Find positions of both entities
    size_t pos_a = text.find(entity_a);
    size_t pos_b = text.find(entity_b);

    if (pos_a == std::string::npos || pos_b == std::string::npos) {
        return relations;
    }

    // Check distance
    size_t dist = (pos_a > pos_b) ? (pos_a - pos_b) : (pos_b - pos_a);
    if (dist > config_.max_entity_distance) {
        return relations;
    }

    // Extract text between entities
    size_t start = std::min(pos_a, pos_b);
    size_t end = std::max(pos_a + entity_a.size(), pos_b + entity_b.size());
    std::string between = text.substr(start, end - start);

    // Check for relation keywords in between text
    struct KeywordRelation {
        std::string keyword;
        RelationType type;
        double confidence;
    };

    static const std::vector<KeywordRelation> keywords = {
        {"is a", RelationType::IS_A, 0.75},
        {"type of", RelationType::IS_A, 0.7},
        {"kind of", RelationType::IS_A, 0.7},
        {"has", RelationType::HAS_PROPERTY, 0.6},
        {"contains", RelationType::HAS_PROPERTY, 0.65},
        {"causes", RelationType::CAUSES, 0.75},
        {"leads to", RelationType::CAUSES, 0.7},
        {"results in", RelationType::CAUSES, 0.7},
        {"enables", RelationType::ENABLES, 0.7},
        {"allows", RelationType::ENABLES, 0.65},
        {"part of", RelationType::PART_OF, 0.75},
        {"belongs to", RelationType::PART_OF, 0.7},
        {"similar to", RelationType::SIMILAR_TO, 0.65},
        {"related to", RelationType::SIMILAR_TO, 0.55},
        {"contradicts", RelationType::CONTRADICTS, 0.75},
        {"opposes", RelationType::CONTRADICTS, 0.7},
        {"supports", RelationType::SUPPORTS, 0.7},
        {"before", RelationType::TEMPORAL_BEFORE, 0.5},
        {"precedes", RelationType::TEMPORAL_BEFORE, 0.7},
        {"and", RelationType::SIMILAR_TO, 0.3},
    };

    // Determine source/target based on position
    const std::string& first_entity = (pos_a < pos_b) ? entity_a : entity_b;
    const std::string& second_entity = (pos_a < pos_b) ? entity_b : entity_a;

    for (const auto& kw : keywords) {
        if (between.find(kw.keyword) != std::string::npos && kw.confidence >= config_.min_confidence) {
            relations.emplace_back(first_entity, second_entity, kw.type,
                                  between, kw.confidence);
            break; // One relation per entity pair from proximity
        }
    }

    return relations;
}

std::vector<SuggestedRelation> RelationExtractor::to_suggested_relations(
    const std::vector<ExtractedRelation>& relations)
{
    std::vector<SuggestedRelation> suggested;
    suggested.reserve(relations.size());

    for (const auto& rel : relations) {
        suggested.emplace_back(
            rel.source_label,
            rel.target_label,
            relation_type_to_str(rel.relation_type),
            rel.evidence_text
        );
    }

    return suggested;
}

std::string RelationExtractor::relation_type_to_str(RelationType type) {
    switch (type) {
        case RelationType::IS_A: return "is-a";
        case RelationType::HAS_PROPERTY: return "has-property";
        case RelationType::CAUSES: return "causes";
        case RelationType::ENABLES: return "enables";
        case RelationType::PART_OF: return "part-of";
        case RelationType::SIMILAR_TO: return "similar-to";
        case RelationType::CONTRADICTS: return "contradicts";
        case RelationType::SUPPORTS: return "supports";
        case RelationType::TEMPORAL_BEFORE: return "temporal-before";
        case RelationType::CUSTOM: return "custom";
        default: return "unknown";
    }
}

} // namespace brain19
