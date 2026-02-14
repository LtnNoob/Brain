#include "semantic_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace brain19 {

// =============================================================================
// SemanticScores
// =============================================================================

size_t SemanticScores::best_template() const {
    if (template_probs.empty()) return 1;  // default: DEFINITIONAL
    return static_cast<size_t>(
        std::distance(template_probs.begin(),
                      std::max_element(template_probs.begin(), template_probs.end())));
}

// =============================================================================
// Construction
// =============================================================================

SemanticScorer::SemanticScorer(const LanguageConfig& config)
    : config_(config)
    , relevance_kan_({32, 16, 1}, config.kan_num_knots)
    , causality_kan_({48, 16, 1}, config.kan_num_knots)
    , template_kan_({16, 8, LanguageConfig::NUM_TEMPLATE_TYPES}, config.kan_num_knots)
{}

// =============================================================================
// Scoring
// =============================================================================

SemanticScores SemanticScorer::score(
    const std::unordered_map<ConceptId, std::vector<double>>& activations,
    const std::vector<double>& query,
    const std::vector<std::pair<ConceptId, ConceptId>>& causal_pairs,
    const std::unordered_map<std::string, std::vector<double>>& relation_embeddings
) const {
    SemanticScores result;

    // Pad/truncate query to 16D for concat
    std::vector<double> q16(16, 0.0);
    for (size_t i = 0; i < std::min(query.size(), size_t(16)); ++i) {
        q16[i] = query[i];
    }

    // Score relevance for each active concept
    for (const auto& [cid, act] : activations) {
        result.relevance[cid] = score_relevance(act, q16);
    }

    // Score causality for each concept pair
    for (const auto& [src, tgt] : causal_pairs) {
        auto src_it = activations.find(src);
        auto tgt_it = activations.find(tgt);
        if (src_it == activations.end() || tgt_it == activations.end()) continue;

        std::string key = std::to_string(src) + ":" + std::to_string(tgt);
        auto rel_it = relation_embeddings.find(key);
        std::vector<double> rel_emb(16, 0.0);
        if (rel_it != relation_embeddings.end()) {
            for (size_t i = 0; i < std::min(rel_it->second.size(), size_t(16)); ++i) {
                rel_emb[i] = rel_it->second[i];
            }
        }

        result.causality[key] = score_causality(src_it->second, tgt_it->second, rel_emb);
    }

    // Classify template type from aggregated activations
    std::vector<double> agg(16, 0.0);
    double n = 0.0;
    for (const auto& [cid, act] : activations) {
        for (size_t i = 0; i < std::min(act.size(), size_t(16)); ++i) {
            agg[i] += act[i];
        }
        n += 1.0;
    }
    if (n > 0) {
        for (auto& v : agg) v /= n;
    }
    result.template_probs = classify_template(agg);

    return result;
}

double SemanticScorer::score_relevance(const std::vector<double>& activation,
                                        const std::vector<double>& query) const {
    // concat(activation[0:16], query[0:16]) → R^32
    std::vector<double> input(32, 0.0);
    for (size_t i = 0; i < std::min(activation.size(), size_t(16)); ++i) {
        input[i] = activation[i];
    }
    for (size_t i = 0; i < std::min(query.size(), size_t(16)); ++i) {
        input[16 + i] = query[i];
    }

    auto out = relevance_kan_.evaluate(input);
    return sigmoid(out.empty() ? 0.0 : out[0]);
}

double SemanticScorer::score_causality(const std::vector<double>& act_source,
                                        const std::vector<double>& act_target,
                                        const std::vector<double>& rel_embedding) const {
    // concat(act_source[0:16], act_target[0:16], rel_embedding[0:16]) → R^48
    std::vector<double> input(48, 0.0);
    for (size_t i = 0; i < std::min(act_source.size(), size_t(16)); ++i) {
        input[i] = act_source[i];
    }
    for (size_t i = 0; i < std::min(act_target.size(), size_t(16)); ++i) {
        input[16 + i] = act_target[i];
    }
    for (size_t i = 0; i < std::min(rel_embedding.size(), size_t(16)); ++i) {
        input[32 + i] = rel_embedding[i];
    }

    auto out = causality_kan_.evaluate(input);
    return sigmoid(out.empty() ? 0.0 : out[0]);
}

std::vector<double> SemanticScorer::classify_template(
    const std::vector<double>& aggregated_activation) const {
    // Pad/truncate to 16D
    std::vector<double> input(16, 0.0);
    for (size_t i = 0; i < std::min(aggregated_activation.size(), size_t(16)); ++i) {
        input[i] = aggregated_activation[i];
    }

    auto logits = template_kan_.evaluate(input);
    return softmax(logits);
}

// =============================================================================
// Helpers
// =============================================================================

double SemanticScorer::sigmoid(double x) {
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<double> SemanticScorer::softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }
    if (sum > 1e-12) {
        for (auto& v : result) v /= sum;
    }
    return result;
}

} // namespace brain19
