#include "kan_encoder.hpp"

#include <algorithm>
#include <cmath>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

KANEncoder::KANEncoder(const LanguageConfig& config)
    : config_(config)
    , kan_({LanguageConfig::TOKEN_EMBED_DIM, 32,
            LanguageConfig::ENCODER_QUERY_DIM, LanguageConfig::ENCODER_QUERY_DIM},
           config.kan_num_knots)
{
    std::mt19937 rng(42);
    init_embeddings(rng);
}

void KANEncoder::init_embeddings(std::mt19937& rng) {
    // Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    double limit = std::sqrt(6.0 / (1.0 + LanguageConfig::TOKEN_EMBED_DIM));
    std::uniform_real_distribution<double> dist(-limit, limit);

    token_embeddings_.resize(LanguageConfig::VOCAB_SIZE);
    for (auto& emb : token_embeddings_) {
        emb.resize(LanguageConfig::TOKEN_EMBED_DIM);
        for (auto& v : emb) {
            v = dist(rng);
        }
    }

    // Zero out special token embeddings (PAD, BOS, EOS, UNK, SEP)
    for (size_t i = 0; i <= LanguageConfig::SEP_TOKEN; ++i) {
        std::fill(token_embeddings_[i].begin(), token_embeddings_[i].end(), 0.0);
    }
}

// =============================================================================
// Encoding
// =============================================================================

std::vector<double> KANEncoder::encode(const std::string& text,
                                         const BPETokenizer& tokenizer) const {
    auto token_ids = tokenizer.encode(text);
    return encode_tokens(token_ids);
}

std::vector<double> KANEncoder::encode_tokens(const std::vector<uint16_t>& token_ids) const {
    if (token_ids.empty()) {
        return std::vector<double>(LanguageConfig::ENCODER_QUERY_DIM, 0.0);
    }

    // Step 1: Bag-of-embeddings with positional decay
    auto x_bag = bag_of_embeddings(token_ids);

    // Step 2: KAN forward pass (64 → 32 → 16 → 16)
    auto query = kan_.evaluate(x_bag);

    // Step 3: L2 normalize output
    l2_normalize(query);

    return query;
}

// =============================================================================
// Bag-of-Embeddings
// =============================================================================

std::vector<double> KANEncoder::bag_of_embeddings(
    const std::vector<uint16_t>& token_ids) const {

    const size_t dim = LanguageConfig::TOKEN_EMBED_DIM;
    std::vector<double> result(dim, 0.0);
    double total_weight = 0.0;

    for (size_t i = 0; i < token_ids.size(); ++i) {
        uint16_t tid = token_ids[i];
        if (tid >= LanguageConfig::VOCAB_SIZE) continue;
        if (tid == LanguageConfig::PAD_TOKEN) continue;

        // Positional decay: content words in the middle get higher weight
        // pos_weight = 1.0 - 0.3 * |2*pos/L - 1| (peaks in center)
        double pos = static_cast<double>(i) / std::max(1.0, static_cast<double>(token_ids.size() - 1));
        double pos_weight = 1.0 - 0.3 * std::abs(2.0 * pos - 1.0);

        const auto& emb = token_embeddings_[tid];
        for (size_t d = 0; d < dim; ++d) {
            result[d] += emb[d] * pos_weight;
        }
        total_weight += pos_weight;
    }

    // Normalize by total weight
    if (total_weight > 1e-12) {
        for (auto& v : result) {
            v /= total_weight;
        }
    }

    // L2 normalize
    l2_normalize(result);

    return result;
}

// =============================================================================
// Helpers
// =============================================================================

const std::vector<double>& KANEncoder::get_token_embedding(uint16_t token_id) const {
    static const std::vector<double> zero(LanguageConfig::TOKEN_EMBED_DIM, 0.0);
    if (token_id < token_embeddings_.size()) {
        return token_embeddings_[token_id];
    }
    return zero;
}

void KANEncoder::set_token_embedding(uint16_t token_id, const std::vector<double>& embedding) {
    if (token_id < token_embeddings_.size()) {
        token_embeddings_[token_id] = embedding;
    }
}

void KANEncoder::l2_normalize(std::vector<double>& vec) {
    double norm = 0.0;
    for (double v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (auto& v : vec) v /= norm;
    }
}

} // namespace brain19
