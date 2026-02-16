#pragma once

#include "language_config.hpp"
#include "bpe_tokenizer.hpp"
#include "fusion_layer.hpp"
#include "../kan/kan_module.hpp"
#include "../micromodel/concept_embedding_store.hpp"
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

struct ConceptDecoderOutput {
    std::vector<ConceptId> concept_ids;    // predicted concept sequence
    std::vector<std::string> labels;       // concept labels
    std::vector<double> confidences;       // per-concept confidence
    std::string text;                      // assembled text from concepts
    double confidence;                     // average confidence
};

class KANDecoder {
public:
    explicit KANDecoder(const LanguageConfig& config = LanguageConfig{});

    // Decode fused representation to text (token-based, backward compat)
    DecoderOutput decode(const FusedRepresentation& fused,
                          const BPETokenizer& tokenizer,
                          const std::vector<std::vector<double>>& token_embeddings,
                          const ConceptEmbeddingStore& concept_embeddings,
                          size_t max_tokens = 30) const;

    // Decode fused representation to concept sequence
    ConceptDecoderOutput decode_concepts(
        const std::vector<double>& extended_fused,
        const ConceptEmbeddingStore& concept_embeddings,
        size_t max_concepts = LanguageConfig::MAX_CONCEPT_SEQUENCE) const;

    // Access KAN modules for training
    KANModule& init_kan() { return init_kan_; }
    KANModule& update_kan() { return update_kan_; }

    // Access output projection for training (token-based, backward compat)
    // Shape: DECODER_HIDDEN_DIM × VOCAB_SIZE (16 × 8192)
    std::vector<std::vector<double>>& output_projection() { return output_projection_; }
    const std::vector<std::vector<double>>& output_projection() const { return output_projection_; }

    // Access concept projection for training: [H × CONCEPT_EMBED_DIM]
    std::vector<std::vector<double>>& concept_projection() { return concept_projection_; }
    const std::vector<std::vector<double>>& concept_projection() const { return concept_projection_; }

    // Re-initialize concept projection for current extended dim
    void reinitialize_concept_projection(size_t H);

    // Set the active concept embedding matrix and IDs for inference.
    // concept_matrix: [N × CONCEPT_EMBED_DIM], concept_ids: [N]
    void set_concept_matrix(std::vector<std::vector<double>> matrix,
                            std::vector<ConceptId> ids,
                            std::vector<std::string> labels);

    // Get decoder confidence threshold
    double confidence_threshold() const { return config_.decoder_confidence_threshold; }

    // Non-linear transform: 2-layer MLP with residual (h' = h + tanh(h·W1+b1)·W2+b2)
    static constexpr size_t TRANSFORM_K = 32;  // bottleneck dim

    // Public accessors for training
    auto& transform_W1() { return transform_W1_; }
    auto& transform_b1() { return transform_b1_; }
    auto& transform_W2() { return transform_W2_; }
    auto& transform_b2() { return transform_b2_; }

    // Forward transform: returns h' = h + tanh(h·W1+b1)·W2+b2
    std::vector<double> transform(const std::vector<double>& h) const;

    // Re-initialize output projection for extended fused dimension.
    // Called after DimensionalContext is built to size the projection
    // for (FUSED_DIM + dim_context_decoder_dim) input.
    void reinitialize_for_extended_dim(size_t extended_fused_dim);
    size_t extended_fused_dim() const { return extended_fused_dim_; }

    // Flexible additional dimensions for v11 dimensional training (default 0 = v10 compat)
    void set_flex_dim(size_t flex_dim) { flex_dim_ = flex_dim; }
    size_t flex_dim() const { return flex_dim_; }

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

    // Extended fused dimension (FUSED_DIM + flex_dim + dimensional context size, runtime)
    size_t extended_fused_dim_ = LanguageConfig::FUSED_DIM;

    // Flexible extra dimensions for v11 (default 0 for backward compat)
    size_t flex_dim_ = 0;

    // Active token set (populated by training, used to suppress untrained tokens)
    std::unordered_set<uint16_t> trained_tokens_;

    // Concept prediction: projection H→16D
    std::vector<std::vector<double>> concept_projection_;  // [H × 16]

    // Cached concept embedding matrix for inference
    std::vector<std::vector<double>> concept_matrix_;      // [N × 16]
    std::vector<ConceptId> concept_ids_;                    // [N]
    std::vector<std::string> concept_labels_;               // [N]

    // Non-linear transform weights (initialized to zero = identity via residual)
    std::vector<std::vector<double>> transform_W1_;  // [H × K]
    std::vector<double> transform_b1_;                // [K]
    std::vector<std::vector<double>> transform_W2_;  // [K × H]
    std::vector<double> transform_b2_;                // [H]

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
