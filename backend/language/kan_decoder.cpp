#include "kan_decoder.hpp"

#include <algorithm>
#include <cmath>
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
    // Initialize output projection: (2*FUSED_DIM) x VOCAB_SIZE (128 x 8192)
    // Uses [h, h^2] quadratic features for non-linear capacity.
    std::mt19937 rng(777);
    double limit = std::sqrt(6.0 / (2 * LanguageConfig::FUSED_DIM + LanguageConfig::VOCAB_SIZE));
    std::uniform_real_distribution<double> dist(-limit, limit);

    output_projection_.resize(2 * LanguageConfig::FUSED_DIM);
    for (auto& row : output_projection_) {
        row.resize(LanguageConfig::VOCAB_SIZE);
        for (auto& v : row) v = dist(rng);
    }

    // Initialize transform for identity-at-init with gradient flow:
    // W1 = small random (breaks symmetry so a1 ≠ 0, enabling W2 gradient)
    // W2 = 0 (identity: h' = h + tanh(h·W1+b1)·0 = h)
    // b1 = 0, b2 = 0
    const size_t H0 = LanguageConfig::FUSED_DIM;
    const size_t K = TRANSFORM_K;
    double w1_scale = 0.01 * std::sqrt(6.0 / (H0 + K));
    std::mt19937 rng_t(42);
    std::uniform_real_distribution<double> dist_t(-w1_scale, w1_scale);
    transform_W1_.resize(H0, std::vector<double>(K));
    for (auto& row : transform_W1_)
        for (auto& v : row) v = dist_t(rng_t);
    transform_b1_.assign(K, 0.0);
    transform_W2_.assign(K, std::vector<double>(H0, 0.0));
    transform_b2_.assign(H0, 0.0);
}

// =============================================================================
// Concept Projection Initialization
// =============================================================================

void KANDecoder::reinitialize_concept_projection(size_t H) {
    const size_t D = LanguageConfig::CONCEPT_EMBED_DIM;  // 16
    std::mt19937 rng(555);
    double limit = std::sqrt(6.0 / (H + D));  // Xavier
    std::uniform_real_distribution<double> dist(-limit, limit);

    concept_projection_.resize(H);
    for (auto& row : concept_projection_) {
        row.resize(D);
        for (auto& v : row) v = dist(rng);
    }
}

void KANDecoder::set_concept_matrix(std::vector<std::vector<double>> matrix,
                                     std::vector<ConceptId> ids,
                                     std::vector<std::string> labels) {
    concept_matrix_ = std::move(matrix);
    concept_ids_ = std::move(ids);
    concept_labels_ = std::move(labels);
}

// =============================================================================
// Re-initialize for extended dimension (after DimensionalContext is built)
// =============================================================================

void KANDecoder::reinitialize_for_extended_dim(size_t extended_fused_dim) {
    extended_fused_dim_ = extended_fused_dim;

    std::mt19937 rng(777);
    double limit = std::sqrt(6.0 / (2 * extended_fused_dim_ + LanguageConfig::VOCAB_SIZE));
    std::uniform_real_distribution<double> dist(-limit, limit);

    output_projection_.resize(2 * extended_fused_dim_);
    for (auto& row : output_projection_) {
        row.assign(LanguageConfig::VOCAB_SIZE, 0.0);
        for (auto& v : row) v = dist(rng);
    }

    // Re-initialize transform: W1 small random (gradient flow), W2=0 (identity)
    const size_t K = TRANSFORM_K;
    double w1_scale = 0.01 * std::sqrt(6.0 / (extended_fused_dim_ + K));
    std::mt19937 rng_t(42);
    std::uniform_real_distribution<double> dist_t(-w1_scale, w1_scale);
    transform_W1_.resize(extended_fused_dim_, std::vector<double>(K));
    for (auto& row : transform_W1_)
        for (auto& v : row) v = dist_t(rng_t);
    transform_b1_.assign(K, 0.0);
    transform_W2_.assign(K, std::vector<double>(extended_fused_dim_, 0.0));
    transform_b2_.assign(extended_fused_dim_, 0.0);
}

// =============================================================================
// Non-linear transform: h' = h + tanh(h·W1+b1)·W2+b2
// =============================================================================

std::vector<double> KANDecoder::transform(const std::vector<double>& h) const {
    const size_t H = h.size();
    const size_t K = TRANSFORM_K;

    // z1 = h · W1 + b1  [K]
    // a1 = tanh(z1)      [K]
    std::vector<double> a1(K);
    for (size_t k = 0; k < K; ++k) {
        double sum = transform_b1_[k];
        for (size_t i = 0; i < H && i < transform_W1_.size(); ++i) {
            sum += h[i] * transform_W1_[i][k];
        }
        a1[k] = std::tanh(sum);
    }

    // h' = h + a1 · W2 + b2  [H]
    std::vector<double> h_out(H);
    for (size_t j = 0; j < H; ++j) {
        double sum = h[j] + transform_b2_[j];
        for (size_t k = 0; k < K && k < transform_W2_.size(); ++k) {
            sum += a1[k] * transform_W2_[k][j];
        }
        h_out[j] = sum;
    }

    return h_out;
}

// =============================================================================
// Concept Decoding
// =============================================================================

ConceptDecoderOutput KANDecoder::decode_concepts(
    const std::vector<double>& extended_fused,
    const ConceptEmbeddingStore& concept_embeddings,
    size_t max_concepts
) const {
    ConceptDecoderOutput result;
    result.confidence = 0.0;

    if (extended_fused.empty() || concept_matrix_.empty()) {
        return result;
    }

    const size_t H = extended_fused_dim_;
    const size_t D = LanguageConfig::CONCEPT_EMBED_DIM;  // 16
    const size_t N = concept_matrix_.size();  // number of active concepts
    const double temperature = config_.concept_inference_temperature;

    // Initialize hidden state from extended fused vector
    std::vector<double> hidden(H, 0.0);
    for (size_t i = 0; i < std::min(H, extended_fused.size()); ++i) {
        hidden[i] = extended_fused[i];
    }

    double total_conf = 0.0;
    std::unordered_set<ConceptId> predicted_set;  // avoid repeats

    for (size_t step = 0; step < max_concepts; ++step) {
        // Apply non-linear transform
        auto h_transformed = transform(hidden);

        // Project to concept embedding space: concept_emb = W_concept^T · h  [D]
        std::vector<double> concept_emb(D, 0.0);
        size_t proj_rows = std::min(h_transformed.size(), concept_projection_.size());
        for (size_t i = 0; i < proj_rows; ++i) {
            for (size_t d = 0; d < D; ++d) {
                concept_emb[d] += h_transformed[i] * concept_projection_[i][d];
            }
        }

        // L2-normalize concept_emb
        double norm_emb = 0.0;
        for (double v : concept_emb) norm_emb += v * v;
        norm_emb = std::sqrt(norm_emb);
        if (norm_emb > 1e-12) {
            for (auto& v : concept_emb) v /= norm_emb;
        }

        // Compute cosine similarity to all concepts in matrix
        std::vector<double> logits(N);
        for (size_t c = 0; c < N; ++c) {
            double dot = 0.0, norm_c = 0.0;
            for (size_t d = 0; d < D; ++d) {
                dot += concept_emb[d] * concept_matrix_[c][d];
                norm_c += concept_matrix_[c][d] * concept_matrix_[c][d];
            }
            norm_c = std::sqrt(norm_c);
            logits[c] = (norm_c > 1e-12) ? (dot / norm_c) : 0.0;

            // Mask already-predicted concepts
            if (predicted_set.count(concept_ids_[c])) {
                logits[c] = -1e9;
            }
        }

        // Temperature-scaled softmax
        std::vector<double> scaled_logits(N);
        for (size_t c = 0; c < N; ++c) {
            scaled_logits[c] = logits[c] / temperature;
        }
        auto probs = softmax(scaled_logits);

        // Argmax
        size_t best_idx = 0;
        double best_prob = probs[0];
        for (size_t c = 1; c < N; ++c) {
            if (probs[c] > best_prob) {
                best_prob = probs[c];
                best_idx = c;
            }
        }

        total_conf += best_prob;

        // Stop if confidence drops too low (after generating at least 1 concept)
        if (step >= 1 && best_prob < config_.decoder_confidence_threshold) {
            break;
        }

        ConceptId pred_cid = concept_ids_[best_idx];
        result.concept_ids.push_back(pred_cid);
        result.labels.push_back(concept_labels_[best_idx]);
        result.confidences.push_back(best_prob);
        predicted_set.insert(pred_cid);

        // Hidden evolution with concept embedding feedback
        auto pred_emb = concept_embeddings.get_or_default(pred_cid);

        // Block 1: Dims 0..63 — concept core projected to 64D
        {
            for (size_t i = 0; i < LanguageConfig::FUSED_DIM && i < CORE_DIM; ++i) {
                hidden[i] = hidden[i] * 0.8 + pred_emb.core[i] * 0.2;
            }
        }

        // Block 2: Dims 64..64+flex_dim-1 — flex detail
        if (flex_dim_ > 0) {
            size_t flex_end = std::min(LanguageConfig::FUSED_DIM + flex_dim_, H);
            for (size_t i = LanguageConfig::FUSED_DIM; i < flex_end; ++i) {
                size_t detail_idx = i - LanguageConfig::FUSED_DIM;
                double flex_val = (detail_idx < pred_emb.detail.size()) ? pred_emb.detail[detail_idx] : 0.0;
                hidden[i] = hidden[i] * 0.9 + flex_val * 0.1;
            }
        }

        // Block 3: DimCtx dims — slow decay
        for (size_t i = LanguageConfig::FUSED_DIM + flex_dim_; i < H; ++i) {
            hidden[i] *= 0.95;
        }
    }

    // Build text from concept sequence + relation verbs
    if (!result.labels.empty()) {
        result.text = result.labels[0];
        for (size_t i = 1; i < result.labels.size(); ++i) {
            result.text += ", " + result.labels[i];
        }
    }

    result.confidence = result.concept_ids.empty() ? 0.0
        : total_conf / static_cast<double>(result.concept_ids.size());

    return result;
}

// =============================================================================
// Helpers
// =============================================================================

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
