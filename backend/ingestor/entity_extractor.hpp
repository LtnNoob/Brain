#pragma once

#include "../importers/knowledge_proposal.hpp"
#include "text_chunker.hpp"
#include <string>
#include <vector>
#include <set>

namespace brain19 {

// ExtractedEntity: An entity found in text with evidence
struct ExtractedEntity {
    std::string label;
    std::string context_snippet;    // Surrounding text for evidence
    size_t frequency;               // How often it appeared
    bool is_capitalized;            // Detected via capitalization pattern
    bool is_quoted;                 // Detected via quotation marks
    bool is_defined;                // Found in "X is ..." pattern

    ExtractedEntity()
        : frequency(1), is_capitalized(false), is_quoted(false), is_defined(false) {}
    ExtractedEntity(const std::string& lbl, const std::string& ctx)
        : label(lbl), context_snippet(ctx), frequency(1),
          is_capitalized(false), is_quoted(false), is_defined(false) {}
};

// EntityExtractor: Pattern-based entity extraction from text
//
// DESIGN:
// - No external NLP dependencies
// - Multiple extraction strategies (capitalization, quotation, definition patterns)
// - Deduplication with frequency counting
// - Configurable extraction limits
// - Returns ExtractedEntity which maps to SuggestedConcept for proposals
//
// EXTRACTION PATTERNS:
// 1. Capitalized multi-word phrases (proper nouns): "Albert Einstein", "Machine Learning"
// 2. Quoted terms: "photosynthesis", 'quantum entanglement'
// 3. Definition patterns: "X is a ...", "X refers to ..."
// 4. Domain terms: Repeated significant words (frequency-based)
class EntityExtractor {
public:
    struct Config {
        size_t max_entities = 50;           // Maximum entities to extract
        size_t min_label_length = 2;        // Minimum entity label length
        size_t max_label_length = 100;      // Maximum entity label length
        size_t context_window = 40;         // Chars before/after for context
        size_t min_frequency_for_common = 2; // Min frequency for common-word entities
        bool extract_capitalized = true;
        bool extract_quoted = true;
        bool extract_defined = true;
        bool extract_frequent = true;
    };

    EntityExtractor() : config_() {}
    explicit EntityExtractor(const Config& config);

    // Extract entities from a single chunk
    std::vector<ExtractedEntity> extract_from_chunk(const TextChunk& chunk) const;

    // Extract entities from multiple chunks, deduplicating
    std::vector<ExtractedEntity> extract_from_chunks(const std::vector<TextChunk>& chunks) const;

    // Extract entities from raw text
    std::vector<ExtractedEntity> extract_from_text(const std::string& text) const;

    // Convert to SuggestedConcept (for KnowledgeProposal compatibility)
    static std::vector<SuggestedConcept> to_suggested_concepts(
        const std::vector<ExtractedEntity>& entities);

    const Config& get_config() const { return config_; }

private:
    Config config_;

    // Extraction strategies
    std::vector<ExtractedEntity> extract_capitalized_phrases(const std::string& text) const;
    std::vector<ExtractedEntity> extract_quoted_terms(const std::string& text) const;
    std::vector<ExtractedEntity> extract_defined_terms(const std::string& text) const;
    std::vector<ExtractedEntity> extract_frequent_terms(const std::string& text) const;

    // Helpers
    std::string get_context(const std::string& text, size_t pos, size_t len) const;
    void deduplicate(std::vector<ExtractedEntity>& entities) const;
    bool is_stopword(const std::string& word) const;
    std::string normalize_label(const std::string& label) const;
};

} // namespace brain19
