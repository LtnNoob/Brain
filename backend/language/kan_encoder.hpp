#pragma once

#include "language_config.hpp"
#include "bpe_tokenizer.hpp"
#include "../kan/kan_module.hpp"
#include <string>
#include <vector>
#include <random>

namespace brain19 {

// =============================================================================
// KAN ENCODER: Text → Query Embedding q ∈ R^16
// =============================================================================
//
// Pipeline:
//   1. Tokenize(text) → token_ids
//   2. Lookup: token_ids → embeddings ∈ R^(L×64)
//   3. Bag-of-Embeddings with Positional Decay: x_bag ∈ R^64
//   4. KAN Layer 1: R^64 → R^32
//   5. KAN Layer 2: R^32 → R^16
//   6. KAN Layer 3: R^16 → R^16 (refinement)
//
// Parameters: Token embeddings (524,288) + KAN layers (28,160) = 552,448
//

class KANEncoder {
public:
    explicit KANEncoder(const LanguageConfig& config = LanguageConfig{});

    // Encode text to query embedding (requires tokenizer)
    std::vector<double> encode(const std::string& text,
                                const BPETokenizer& tokenizer) const;

    // Encode from pre-tokenized IDs
    std::vector<double> encode_tokens(const std::vector<uint16_t>& token_ids) const;

    // Get token embedding by ID
    const std::vector<double>& get_token_embedding(uint16_t token_id) const;

    // Set token embedding (for training)
    void set_token_embedding(uint16_t token_id, const std::vector<double>& embedding);

    // Access KAN module for training
    KANModule& kan_module() { return kan_; }
    const KANModule& kan_module() const { return kan_; }

    // Access embeddings table for training
    std::vector<std::vector<double>>& embedding_table() { return token_embeddings_; }
    const std::vector<std::vector<double>>& embedding_table() const { return token_embeddings_; }

    size_t embedding_dim() const { return LanguageConfig::TOKEN_EMBED_DIM; }
    size_t output_dim() const { return LanguageConfig::ENCODER_QUERY_DIM; }

private:
    LanguageConfig config_;

    // Token embedding table: VOCAB_SIZE × TOKEN_EMBED_DIM (8192 × 64 = 524,288 params)
    std::vector<std::vector<double>> token_embeddings_;

    // KAN: 64 → 32 → 16 → 16 (28,160 params)
    KANModule kan_;

    // Bag-of-embeddings with positional decay
    std::vector<double> bag_of_embeddings(const std::vector<uint16_t>& token_ids) const;

    // L2 normalize a vector in-place
    static void l2_normalize(std::vector<double>& vec);

    // Initialize embeddings with Xavier/Glorot uniform
    void init_embeddings(std::mt19937& rng);
};

} // namespace brain19
