#pragma once

#include "language_config.hpp"
#include "bpe_tokenizer.hpp"
#include "fusion_layer.hpp"
#include "../kan/kan_module.hpp"
#include <string>
#include <unordered_set>
#include <vector>

namespace brain19 {

// =============================================================================
// KAN DECODER: Fused Representation → Token Sequence
// =============================================================================
//
// Autoregressive generation:
//   1. h₀ = KAN_init(fused_vector):  R^64 → R^16
//   2. Loop:
//      a. logits = h_t · E_vocab^T ∈ R^8192
//      b. Constrained sampling (concept tokens boosted)
//      c. token_t = argmax(logits)
//      d. h_{t+1} = KAN_update(concat(h_t, embed(token_t))):  R^80 → R^32 → R^16
//   3. Stop at EOS or max_tokens
//
// Parameters:
//   Init KAN: 64→16 = 10,240
//   Update KAN: 80→32→16 = 30,720
//   Output projection: 16 × 8192 = 131,072
//   Total: 172,032
//

struct DecoderOutput {
    std::vector<uint16_t> token_ids;
    std::string text;
    double confidence;    // average max-softmax across generated tokens
    bool used_template;   // true if confidence dropped below threshold
};

class KANDecoder {
public:
    explicit KANDecoder(const LanguageConfig& config = LanguageConfig{});

    // Decode fused representation to text
    DecoderOutput decode(const FusedRepresentation& fused,
                          const BPETokenizer& tokenizer,
                          const std::vector<std::vector<double>>& token_embeddings,
                          size_t max_tokens = 30) const;

    // Access KAN modules for training
    KANModule& init_kan() { return init_kan_; }
    KANModule& update_kan() { return update_kan_; }

    // Access output projection for training
    // Shape: DECODER_HIDDEN_DIM × VOCAB_SIZE (16 × 8192)
    std::vector<std::vector<double>>& output_projection() { return output_projection_; }
    const std::vector<std::vector<double>>& output_projection() const { return output_projection_; }

    // Get decoder confidence threshold
    double confidence_threshold() const { return config_.decoder_confidence_threshold; }

    // Set the active token vocabulary (tokens trained by decoder training)
    // At inference, all other tokens are suppressed to -inf logits.
    void set_trained_tokens(const std::vector<uint16_t>& tokens) {
        trained_tokens_.clear();
        trained_tokens_.insert(tokens.begin(), tokens.end());
    }

private:
    LanguageConfig config_;

    // Init KAN: R^64 → R^16
    KANModule init_kan_;

    // Update KAN: R^80 → R^32 → R^16
    KANModule update_kan_;

    // Output projection: R^FUSED_DIM → R^VOCAB_SIZE
    std::vector<std::vector<double>> output_projection_;

    // Active token set (populated by training, used to suppress untrained tokens)
    std::unordered_set<uint16_t> trained_tokens_;

    // Compute logits from hidden state
    std::vector<double> compute_logits(const std::vector<double>& hidden) const;

    // Apply concept token boost
    void boost_concept_tokens(std::vector<double>& logits,
                               const std::unordered_set<ConceptId>& active_concepts,
                               const BPETokenizer& tokenizer) const;

    // Softmax
    static std::vector<double> softmax(const std::vector<double>& logits);
};

} // namespace brain19
