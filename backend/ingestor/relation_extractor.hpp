#pragma once

#include "../importers/knowledge_proposal.hpp"
#include "../memory/active_relation.hpp"
#include "text_chunker.hpp"
#include "entity_extractor.hpp"
#include <string>
#include <vector>

namespace brain19 {

// ExtractedRelation: A relation found between entities in text
struct ExtractedRelation {
    std::string source_label;
    std::string target_label;
    RelationType relation_type;
    std::string evidence_text;      // The text that evidences this relation
    double confidence;              // [0.0, 1.0] based on pattern strength

    ExtractedRelation()
        : relation_type(RelationType::CUSTOM), confidence(0.5) {}
    ExtractedRelation(const std::string& src, const std::string& tgt,
                     RelationType type, const std::string& evidence, double conf = 0.5)
        : source_label(src), target_label(tgt), relation_type(type),
          evidence_text(evidence), confidence(conf) {}
};

// RelationExtractor: Pattern-based relation extraction from text
//
// DESIGN:
// - No external NLP dependencies
// - Pattern-matching for common relation types
// - Works with already-extracted entities for targeted extraction
// - Maps patterns to existing brain19::RelationType enum
//
// RELATION PATTERNS (mapped to RelationType):
// - "X is a Y"           → IS_A
// - "X has Y"            → HAS_PROPERTY
// - "X causes Y"         → CAUSES
// - "X enables Y"        → ENABLES
// - "X is part of Y"     → PART_OF
// - "X is similar to Y"  → SIMILAR_TO
// - "X contradicts Y"    → CONTRADICTS
// - "X supports Y"       → SUPPORTS
// - "X before Y"         → TEMPORAL_BEFORE
class RelationExtractor {
public:
    struct Config {
        size_t max_relations = 50;
        double min_confidence = 0.3;
        size_t max_entity_distance = 200;  // Max chars between entities for relation
    };

    RelationExtractor() : config_() { init_patterns(); }
    explicit RelationExtractor(const Config& config);

    // Extract relations from text using known entities
    std::vector<ExtractedRelation> extract_relations(
        const std::string& text,
        const std::vector<ExtractedEntity>& known_entities) const;

    // Extract relations from text without pre-extracted entities
    std::vector<ExtractedRelation> extract_relations_blind(const std::string& text) const;

    // Convert to SuggestedRelation (for KnowledgeProposal compatibility)
    static std::vector<SuggestedRelation> to_suggested_relations(
        const std::vector<ExtractedRelation>& relations);

    // Convert RelationType to string for SuggestedRelation
    static std::string relation_type_to_str(RelationType type);

    const Config& get_config() const { return config_; }

private:
    Config config_;

    // Pattern definition
    struct RelationPattern {
        std::string pattern_str;    // Regex pattern with capture groups
        RelationType type;
        double base_confidence;
        bool source_is_group1;      // true: group1=source, group2=target
    };

    std::vector<RelationPattern> patterns_;

    void init_patterns();

    // Entity-aware extraction: find relations between known entities
    std::vector<ExtractedRelation> extract_entity_pair_relations(
        const std::string& text,
        const std::string& entity_a,
        const std::string& entity_b) const;
};

} // namespace brain19
