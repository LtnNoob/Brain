#include "entity_extractor.hpp"
#include <algorithm>
#include <cctype>
#include <regex>
#include <map>
#include <sstream>

namespace brain19 {

EntityExtractor::EntityExtractor(const Config& config)
    : config_(config)
{
}

std::vector<ExtractedEntity> EntityExtractor::extract_from_text(const std::string& text) const {
    std::vector<ExtractedEntity> all_entities;

    if (config_.extract_capitalized) {
        auto caps = extract_capitalized_phrases(text);
        all_entities.insert(all_entities.end(), caps.begin(), caps.end());
    }

    if (config_.extract_quoted) {
        auto quoted = extract_quoted_terms(text);
        all_entities.insert(all_entities.end(), quoted.begin(), quoted.end());
    }

    if (config_.extract_defined) {
        auto defined = extract_defined_terms(text);
        all_entities.insert(all_entities.end(), defined.begin(), defined.end());
    }

    if (config_.extract_frequent) {
        auto frequent = extract_frequent_terms(text);
        all_entities.insert(all_entities.end(), frequent.begin(), frequent.end());
    }

    deduplicate(all_entities);

    // Sort by frequency (descending), then alphabetically
    std::sort(all_entities.begin(), all_entities.end(),
        [](const ExtractedEntity& a, const ExtractedEntity& b) {
            if (a.frequency != b.frequency) return a.frequency > b.frequency;
            return a.label < b.label;
        });

    // Enforce limit
    if (all_entities.size() > config_.max_entities) {
        all_entities.resize(config_.max_entities);
    }

    return all_entities;
}

std::vector<ExtractedEntity> EntityExtractor::extract_from_chunk(const TextChunk& chunk) const {
    return extract_from_text(chunk.text);
}

std::vector<ExtractedEntity> EntityExtractor::extract_from_chunks(
    const std::vector<TextChunk>& chunks) const
{
    // Merge all chunk texts for global extraction
    std::string combined;
    for (const auto& chunk : chunks) {
        if (!combined.empty()) combined += " ";
        combined += chunk.text;
    }
    return extract_from_text(combined);
}

std::vector<ExtractedEntity> EntityExtractor::extract_capitalized_phrases(
    const std::string& text) const
{
    std::vector<ExtractedEntity> entities;

    // Match capitalized word sequences (2+ chars, not at sentence start)
    // Pattern: One or more capitalized words in sequence
    std::regex cap_regex("\\b([A-Z][a-zA-Z]*(?:\\s+[A-Z][a-zA-Z]*)*)\\b");
    std::smatch match;

    std::string::const_iterator search_start = text.cbegin();
    while (std::regex_search(search_start, text.cend(), match, cap_regex)) {
        std::string label = match[1].str();

        // Skip if too short or too long
        if (label.size() < config_.min_label_length || label.size() > config_.max_label_length) {
            search_start = match.suffix().first;
            continue;
        }

        // Skip if it's a sentence-start word (preceded by ". " or start of text)
        size_t pos = static_cast<size_t>(match.position()) +
                     static_cast<size_t>(std::distance(text.cbegin(), search_start));

        bool is_sentence_start = (pos == 0);
        if (!is_sentence_start && pos >= 2) {
            // Check if preceded by sentence-ending punctuation + space
            char prev1 = text[pos - 1];
            char prev2 = text[pos - 2];
            if ((prev2 == '.' || prev2 == '!' || prev2 == '?') && prev1 == ' ') {
                is_sentence_start = true;
            }
        }

        // Only skip single-word sentence starters; multi-word caps are likely entities
        bool is_multi_word = label.find(' ') != std::string::npos;
        if (is_sentence_start && !is_multi_word) {
            search_start = match.suffix().first;
            continue;
        }

        // Skip common non-entity capitalized words
        if (!is_multi_word && is_stopword(label)) {
            search_start = match.suffix().first;
            continue;
        }

        ExtractedEntity entity(label, get_context(text, pos, label.size()));
        entity.is_capitalized = true;
        entities.push_back(entity);

        search_start = match.suffix().first;
        if (entities.size() >= config_.max_entities * 2) break;
    }

    return entities;
}

std::vector<ExtractedEntity> EntityExtractor::extract_quoted_terms(
    const std::string& text) const
{
    std::vector<ExtractedEntity> entities;

    // Match double-quoted and single-quoted terms
    std::regex quote_regex("[\"']([^\"']{2,80})[\"']");
    std::smatch match;

    std::string::const_iterator search_start = text.cbegin();
    while (std::regex_search(search_start, text.cend(), match, quote_regex)) {
        std::string label = match[1].str();

        if (label.size() >= config_.min_label_length && label.size() <= config_.max_label_length) {
            size_t pos = static_cast<size_t>(match.position()) +
                         static_cast<size_t>(std::distance(text.cbegin(), search_start));
            ExtractedEntity entity(label, get_context(text, pos, label.size() + 2));
            entity.is_quoted = true;
            entities.push_back(entity);
        }

        search_start = match.suffix().first;
        if (entities.size() >= config_.max_entities) break;
    }

    return entities;
}

std::vector<ExtractedEntity> EntityExtractor::extract_defined_terms(
    const std::string& text) const
{
    std::vector<ExtractedEntity> entities;

    // Patterns: "X is a ...", "X refers to ...", "X is defined as ..."
    std::vector<std::regex> def_patterns = {
        std::regex("([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+is\\s+(?:a|an|the)\\s+"),
        std::regex("([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+refers?\\s+to\\s+"),
        std::regex("([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+is\\s+defined\\s+as\\s+"),
        std::regex("([A-Z][a-zA-Z]*(?:\\s+[a-zA-Z]+)*)\\s+(?:are|were|was)\\s+(?:a|an|the)\\s+"),
    };

    for (const auto& pattern : def_patterns) {
        std::smatch match;
        std::string::const_iterator search_start = text.cbegin();

        while (std::regex_search(search_start, text.cend(), match, pattern)) {
            std::string label = match[1].str();

            if (label.size() >= config_.min_label_length &&
                label.size() <= config_.max_label_length &&
                !is_stopword(label)) {
                size_t pos = static_cast<size_t>(match.position()) +
                             static_cast<size_t>(std::distance(text.cbegin(), search_start));
                ExtractedEntity entity(label, get_context(text, pos, match[0].str().size()));
                entity.is_defined = true;
                entities.push_back(entity);
            }

            search_start = match.suffix().first;
            if (entities.size() >= config_.max_entities) break;
        }
    }

    return entities;
}

std::vector<ExtractedEntity> EntityExtractor::extract_frequent_terms(
    const std::string& text) const
{
    std::vector<ExtractedEntity> entities;

    // Count word frequencies (lowercase, excluding stopwords)
    std::map<std::string, size_t> freq_map;
    std::map<std::string, std::string> original_case; // Keep original casing

    std::regex word_regex("\\b([a-zA-Z]{3,})\\b");
    std::smatch match;
    std::string::const_iterator search_start = text.cbegin();

    while (std::regex_search(search_start, text.cend(), match, word_regex)) {
        std::string word = match[1].str();
        std::string lower = normalize_label(word);

        if (!is_stopword(lower) && lower.size() >= config_.min_label_length) {
            freq_map[lower]++;
            if (original_case.find(lower) == original_case.end()) {
                original_case[lower] = word;
            }
        }

        search_start = match.suffix().first;
    }

    // Extract terms that appear frequently enough
    for (const auto& [lower, count] : freq_map) {
        if (count >= config_.min_frequency_for_common) {
            const std::string& original = original_case[lower];
            // Find first occurrence for context
            size_t pos = text.find(original);
            std::string ctx = (pos != std::string::npos)
                ? get_context(text, pos, original.size())
                : "";

            ExtractedEntity entity(original, ctx);
            entity.frequency = count;
            entities.push_back(entity);
        }
    }

    return entities;
}

std::vector<SuggestedConcept> EntityExtractor::to_suggested_concepts(
    const std::vector<ExtractedEntity>& entities)
{
    std::vector<SuggestedConcept> concepts;
    concepts.reserve(entities.size());

    for (const auto& entity : entities) {
        concepts.emplace_back(entity.label, entity.context_snippet);
    }

    return concepts;
}

std::string EntityExtractor::get_context(
    const std::string& text, size_t pos, size_t len) const
{
    size_t ctx_start = (pos > config_.context_window) ? (pos - config_.context_window) : 0;
    size_t ctx_end = std::min(pos + len + config_.context_window, text.size());
    return text.substr(ctx_start, ctx_end - ctx_start);
}

void EntityExtractor::deduplicate(std::vector<ExtractedEntity>& entities) const {
    // Group by normalized label, merge properties
    std::map<std::string, size_t> label_index; // normalized → index in result

    std::vector<ExtractedEntity> deduped;

    for (auto& entity : entities) {
        std::string key = normalize_label(entity.label);

        auto it = label_index.find(key);
        if (it != label_index.end()) {
            // Merge: increase frequency, keep richer metadata
            auto& existing = deduped[it->second];
            existing.frequency += entity.frequency;
            if (entity.is_capitalized) existing.is_capitalized = true;
            if (entity.is_quoted) existing.is_quoted = true;
            if (entity.is_defined) existing.is_defined = true;
            // Keep the longer context
            if (entity.context_snippet.size() > existing.context_snippet.size()) {
                existing.context_snippet = entity.context_snippet;
            }
        } else {
            label_index[key] = deduped.size();
            deduped.push_back(std::move(entity));
        }
    }

    entities = std::move(deduped);
}

bool EntityExtractor::is_stopword(const std::string& word) const {
    static const std::set<std::string> stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "need",
        "must", "it", "its", "this", "that", "these", "those", "he", "she",
        "they", "we", "you", "who", "which", "what", "where", "when", "how",
        "not", "no", "nor", "if", "then", "than", "too", "very", "also",
        "just", "about", "above", "after", "before", "between", "into",
        "through", "during", "here", "there", "some", "such", "other",
        "each", "every", "all", "both", "few", "more", "most", "only",
        "own", "same", "so", "still", "while", "however", "therefore",
        "The", "This", "That", "These", "Those", "There", "Here",
        "It", "He", "She", "They", "We", "You", "His", "Her", "Its",
        "Our", "Your", "Their", "Some", "Many", "Most", "Each", "Every",
        "Both", "All", "Such", "Other", "Another", "Any"
    };

    std::string lower = normalize_label(word);
    return stopwords.count(lower) > 0 || stopwords.count(word) > 0;
}

std::string EntityExtractor::normalize_label(const std::string& label) const {
    std::string lower;
    lower.reserve(label.size());
    for (char c : label) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return lower;
}

} // namespace brain19
