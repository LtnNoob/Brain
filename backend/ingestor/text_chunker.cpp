#include "text_chunker.hpp"
#include <algorithm>
#include <cctype>

namespace brain19 {

TextChunker::TextChunker(const Config& config)
    : config_(config)
{
}

std::vector<std::string> TextChunker::split_sentences(const std::string& text) const {
    std::vector<std::string> sentences;
    if (text.empty()) return sentences;

    size_t start = 0;

    for (size_t i = 0; i < text.size(); ++i) {
        if (is_sentence_end(text, i)) {
            // Include the punctuation in the sentence
            std::string sentence = trim(text.substr(start, i - start + 1));
            if (!sentence.empty()) {
                sentences.push_back(sentence);
            }
            // Skip whitespace after sentence end
            start = i + 1;
            while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
                ++start;
            }
            i = (start > 0) ? start - 1 : 0;
        }
    }

    // Handle remaining text (no terminating punctuation)
    if (start < text.size()) {
        std::string remainder = trim(text.substr(start));
        if (!remainder.empty()) {
            sentences.push_back(remainder);
        }
    }

    return sentences;
}

std::vector<TextChunk> TextChunker::chunk_text(const std::string& text) const {
    std::vector<TextChunk> chunks;
    if (text.empty()) return chunks;

    auto sentences = split_sentences(text);
    if (sentences.empty()) return chunks;

    size_t step = config_.sentences_per_chunk;
    if (config_.overlap_sentences >= step) {
        // Overlap must be less than chunk size; fallback to no overlap
        step = config_.sentences_per_chunk;
    } else {
        step = config_.sentences_per_chunk - config_.overlap_sentences;
    }
    if (step == 0) step = 1;

    // Track offsets in original text
    std::vector<size_t> sentence_starts;
    std::vector<size_t> sentence_ends;
    size_t search_from = 0;
    for (const auto& sent : sentences) {
        size_t pos = text.find(sent, search_from);
        if (pos == std::string::npos) {
            // Fallback: approximate
            pos = search_from;
        }
        sentence_starts.push_back(pos);
        sentence_ends.push_back(pos + sent.size());
        search_from = pos + sent.size();
    }

    size_t chunk_idx = 0;
    for (size_t i = 0; i < sentences.size(); i += step) {
        size_t end_idx = std::min(i + config_.sentences_per_chunk, sentences.size());

        // Build chunk text
        std::string chunk_text;
        for (size_t j = i; j < end_idx; ++j) {
            if (!chunk_text.empty()) chunk_text += " ";
            chunk_text += sentences[j];
        }

        // Enforce max chunk size
        if (chunk_text.size() > config_.max_chunk_chars) {
            chunk_text = chunk_text.substr(0, config_.max_chunk_chars);
        }

        // Skip tiny chunks
        if (chunk_text.size() < config_.min_chunk_chars && chunk_idx > 0) {
            continue;
        }

        size_t start_off = sentence_starts[i];
        size_t end_off = sentence_ends[end_idx - 1];

        chunks.emplace_back(chunk_text, start_off, end_off, chunk_idx);
        ++chunk_idx;

        // If we've consumed all sentences, stop
        if (end_idx >= sentences.size()) break;
    }

    return chunks;
}

bool TextChunker::is_sentence_end(const std::string& text, size_t pos) const {
    char c = text[pos];

    // Direct sentence terminators
    if (c == '!' || c == '?') {
        return true;
    }

    if (c == '.') {
        // Avoid splitting on abbreviations: "Dr.", "Mr.", "e.g.", "i.e.", etc.
        // Heuristic: period followed by space+uppercase or end of text = sentence end
        // Period followed by lowercase or another period = not sentence end

        // End of text
        if (pos + 1 >= text.size()) return true;

        char next = text[pos + 1];

        // Period followed by whitespace
        if (std::isspace(static_cast<unsigned char>(next))) {
            // Look ahead past whitespace for uppercase
            size_t look = pos + 2;
            while (look < text.size() && std::isspace(static_cast<unsigned char>(text[look]))) {
                ++look;
            }
            if (look >= text.size()) return true;
            // Next non-space is uppercase or digit → sentence end
            if (std::isupper(static_cast<unsigned char>(text[look])) ||
                std::isdigit(static_cast<unsigned char>(text[look]))) {
                // Check for common abbreviations: single uppercase letter before period
                if (pos > 0 && std::isupper(static_cast<unsigned char>(text[pos - 1]))) {
                    // Could be abbreviation like "U.S." - check if preceded by period
                    if (pos >= 2 && text[pos - 2] == '.') {
                        return false; // Likely abbreviation
                    }
                }
                return true;
            }
            // Followed by quote or parenthesis → sentence end
            if (text[look] == '"' || text[look] == '\'' || text[look] == '(') {
                return true;
            }
            return false;
        }

        // Period followed by quote → sentence end
        if (next == '"' || next == '\'') return true;

        // Period followed by newline → sentence end
        if (next == '\n') return true;

        return false;
    }

    return false;
}

std::string TextChunker::trim(const std::string& s) const {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

} // namespace brain19
