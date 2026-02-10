#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace brain19 {

// TextChunk: A segment of text with position metadata
struct TextChunk {
    std::string text;
    size_t start_offset;    // Position in original text
    size_t end_offset;
    size_t chunk_index;     // Sequential index

    TextChunk() : start_offset(0), end_offset(0), chunk_index(0) {}
    TextChunk(const std::string& t, size_t start, size_t end, size_t idx)
        : text(t), start_offset(start), end_offset(end), chunk_index(idx) {}
};

// TextChunker: Splits plain text into sentence-based chunks
//
// DESIGN:
// - Sentence-boundary detection (period, exclamation, question mark)
// - Configurable chunk size (number of sentences per chunk)
// - Optional overlap between chunks for context continuity
// - No external NLP dependencies - uses rule-based splitting
//
// INVARIANT: Original text can be reconstructed from chunks (minus overlap)
class TextChunker {
public:
    struct Config {
        size_t sentences_per_chunk = 3;     // Sentences grouped per chunk
        size_t overlap_sentences = 1;        // Overlap for context continuity
        size_t max_chunk_chars = 2000;       // Hard limit on chunk size
        size_t min_chunk_chars = 20;         // Skip tiny chunks
    };

    TextChunker() : config_() {}
    explicit TextChunker(const Config& config);

    // Split text into chunks
    std::vector<TextChunk> chunk_text(const std::string& text) const;

    // Split into raw sentences first (useful for inspection)
    std::vector<std::string> split_sentences(const std::string& text) const;

    const Config& get_config() const { return config_; }

private:
    Config config_;

    bool is_sentence_end(const std::string& text, size_t pos) const;
    std::string trim(const std::string& s) const;
};

} // namespace brain19
