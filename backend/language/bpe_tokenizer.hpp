#pragma once

#include "language_config.hpp"
#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// BPE TOKENIZER (8K vocabulary)
// =============================================================================
//
// Byte-Pair Encoding tokenizer with concept token mapping.
// Tokens 0-4: special tokens (PAD, BOS, EOS, UNK, SEP)
// Tokens 5-1004: concept tokens (mapped to ConceptIds from LTM)
// Tokens 1005-8191: BPE merge tokens
//

class BPETokenizer {
public:
    BPETokenizer();

    // Encode text to token IDs
    std::vector<uint16_t> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<uint16_t>& tokens) const;

    // Train BPE merges from corpus + LTM concept labels
    void train(const std::vector<std::string>& corpus,
               const LongTermMemory& ltm,
               size_t target_vocab_size = LanguageConfig::VOCAB_SIZE);

    // Build vocabulary from LTM only (no corpus needed for basic operation)
    void build_from_ltm(const LongTermMemory& ltm);

    // Concept token mapping
    std::optional<ConceptId> token_to_concept(uint16_t token) const;
    std::optional<uint16_t> concept_to_token(ConceptId cid) const;

    // Check if token is a concept token
    bool is_concept_token(uint16_t token) const;

    size_t vocab_size() const { return id_to_token_.size(); }
    bool is_trained() const { return !id_to_token_.empty(); }

    // Get token embedding index for lookup
    uint16_t get_token_id(const std::string& token) const;

    // Word-level tokenization (whitespace + punctuation split)
    // Exposes word boundaries for sentence parsing without BPE subword merging
    std::vector<std::string> word_tokenize(const std::string& text) const;

    // Persistence
    void save(const std::string& path) const;
    bool load(const std::string& path);

private:
    // BPE merge rules: ordered pairs of strings to merge
    std::vector<std::pair<std::string, std::string>> merges_;

    // Token <-> ID mapping
    std::unordered_map<std::string, uint16_t> token_to_id_;
    std::vector<std::string> id_to_token_;

    // Concept <-> token mapping
    std::unordered_map<uint16_t, ConceptId> token_concept_map_;
    std::unordered_map<ConceptId, uint16_t> concept_token_map_;

    // Initialize base vocabulary (256 byte tokens + special tokens)
    void init_base_vocab();

    // Apply BPE merges to a word
    std::vector<std::string> bpe_encode_word(const std::string& word) const;

    // Split text into words (whitespace-based with punctuation separation)
    std::vector<std::string> split_words(const std::string& text) const;

    // Count pair frequencies in a tokenized corpus
    using PairCount = std::unordered_map<std::string, size_t>;
    PairCount count_pairs(const std::vector<std::vector<std::string>>& tokenized) const;
};

} // namespace brain19
