#include "kan_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

KANDecoder::KANDecoder(const LanguageConfig& config)
    : config_(config)
    , init_kan_({LanguageConfig::FUSED_DIM, LanguageConfig::DECODER_HIDDEN_DIM},
                config.kan_num_knots)
    , update_kan_({LanguageConfig::DECODER_HIDDEN_DIM + LanguageConfig::TOKEN_EMBED_DIM,
                   32, LanguageConfig::DECODER_HIDDEN_DIM},
                  config.kan_num_knots)
{
    // Initialize output projection: (2*FUSED_DIM) × VOCAB_SIZE (128 × 8192)
    // Uses [h, h²] quadratic features for non-linear capacity.
    std::mt19937 rng(777);
    double limit = std::sqrt(6.0 / (2 * LanguageConfig::FUSED_DIM + LanguageConfig::VOCAB_SIZE));
    std::uniform_real_distribution<double> dist(-limit, limit);

    output_projection_.resize(2 * LanguageConfig::FUSED_DIM);
    for (auto& row : output_projection_) {
        row.resize(LanguageConfig::VOCAB_SIZE);
        for (auto& v : row) v = dist(rng);
    }
}

// =============================================================================
// Decoding
// =============================================================================

DecoderOutput KANDecoder::decode(
    const FusedRepresentation& fused,
    const BPETokenizer& tokenizer,
    const std::vector<std::vector<double>>& token_embeddings,
    size_t max_tokens
) const {
    DecoderOutput result;
    result.used_template = false;
    result.confidence = 0.0;

    if (fused.fused_vector.empty()) {
        result.used_template = true;
        result.confidence = 0.0;
        return result;
    }

    // Build set of active concept IDs for boosting
    std::unordered_set<ConceptId> active_concepts(
        fused.ordered_concepts.begin(), fused.ordered_concepts.end());

    // Step 0: Initialize hidden state from full fused vector (64D)
    // Bypass init_kan — training uses fused directly, inference must match.
    const size_t H = LanguageConfig::FUSED_DIM;  // 64
    std::vector<double> hidden(H, 0.0);
    for (size_t i = 0; i < std::min(H, fused.fused_vector.size()); ++i) {
        hidden[i] = fused.fused_vector[i];
    }

    double total_conf = 0.0;
    size_t generated = 0;

    for (size_t t = 0; t < max_tokens; ++t) {
        // Build quadratic features: [h, h²] for non-linear capacity
        std::vector<double> h_ext(2 * H);
        for (size_t i = 0; i < H; ++i) {
            h_ext[i] = hidden[i];
            h_ext[H + i] = hidden[i] * hidden[i];
        }

        // Compute logits from extended features
        auto logits = compute_logits(h_ext);

        // Boost concept tokens
        boost_concept_tokens(logits, active_concepts, tokenizer);

        // Suppress special tokens (PAD, BOS, UNK, SEP)
        logits[LanguageConfig::PAD_TOKEN] = -1e9;
        logits[LanguageConfig::BOS_TOKEN] = -1e9;
        logits[LanguageConfig::UNK_TOKEN] = -1e9;
        logits[LanguageConfig::SEP_TOKEN] = -1e9;

        // Suppress all non-trained tokens (they have random W, drowning softmax)
        if (!trained_tokens_.empty()) {
            for (size_t v = 0; v < logits.size(); ++v) {
                if (trained_tokens_.find(static_cast<uint16_t>(v)) == trained_tokens_.end()) {
                    logits[v] = -1e9;
                }
            }
        }

        // Softmax
        auto probs = softmax(logits);

        // Greedy: argmax
        auto max_it = std::max_element(probs.begin(), probs.end());
        size_t best_idx = static_cast<size_t>(std::distance(probs.begin(), max_it));
        double best_prob = *max_it;

        // Check confidence
        total_conf += best_prob;
        generated++;

        // If average confidence drops too low, trigger template fallback
        if (generated >= 3 && (total_conf / generated) < config_.decoder_confidence_threshold) {
            result.used_template = true;
            break;
        }

        uint16_t token = static_cast<uint16_t>(best_idx);

        // Stop at EOS or period
        if (token == LanguageConfig::EOS_TOKEN) break;

        result.token_ids.push_back(token);

        // Check if we generated a period and already have content
        auto decoded_tok = (token < tokenizer.vocab_size()) ?
            std::string(1, '.') : "";  // placeholder
        // Simple: if token is ".", stop after generating it
        if (token < token_embeddings.size()) {
            // Get the token string for stop detection
            auto decoded_tokens = tokenizer.decode({token});
            if (decoded_tokens == "." && result.token_ids.size() > 3) {
                break;
            }
        }

        // Update hidden state: simple linear mixing (matches training dynamics)
        // Bypass update_kan — cheap O(H) update gives position-dependent outputs
        if (token < token_embeddings.size()) {
            const auto& emb = token_embeddings[token];
            for (size_t i = 0; i < H && i < emb.size(); ++i) {
                hidden[i] = hidden[i] * 0.8 + emb[i] * 0.2;
            }
        }
    }

    // Decode tokens to text
    result.text = tokenizer.decode(result.token_ids);
    result.confidence = (generated > 0) ? (total_conf / generated) : 0.0;

    return result;
}

// =============================================================================
// Helpers
// =============================================================================

std::vector<double> KANDecoder::compute_logits(const std::vector<double>& hidden) const {
    // h · W^T = logits ∈ R^VOCAB_SIZE
    std::vector<double> logits(LanguageConfig::VOCAB_SIZE, 0.0);
    size_t h_dim = std::min(hidden.size(), output_projection_.size());

    for (size_t i = 0; i < h_dim; ++i) {
        for (size_t v = 0; v < LanguageConfig::VOCAB_SIZE; ++v) {
            logits[v] += hidden[i] * output_projection_[i][v];
        }
    }
    return logits;
}

void KANDecoder::boost_concept_tokens(
    std::vector<double>& logits,
    const std::unordered_set<ConceptId>& active_concepts,
    const BPETokenizer& tokenizer
) const {
    // Boost tokens corresponding to active concepts
    for (auto cid : active_concepts) {
        auto tok_opt = tokenizer.concept_to_token(cid);
        if (tok_opt && *tok_opt < logits.size()) {
            logits[*tok_opt] += config_.concept_token_boost;
        }
    }
}

std::vector<double> KANDecoder::softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(std::min(logits[i] - max_val, 80.0));  // prevent overflow
        sum += result[i];
    }
    if (sum > 1e-12) {
        for (auto& v : result) v /= sum;
    }
    return result;
}

} // namespace brain19
