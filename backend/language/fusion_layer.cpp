#include "fusion_layer.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

FusionLayer::FusionLayer(const LanguageConfig& config)
    : config_(config)
{
    // Initialize gate weights with small values
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 0.1);

    gate_w_.resize(LanguageConfig::GATE_INPUT_DIM);
    for (auto& w : gate_w_) w = dist(rng);
    gate_b_ = 0.0;

    // Initialize projection matrix: 57 → 64
    // raw_dim = TOP_K_CONCEPTS * 16 + max(TOP_K_CONCEPTS+2, 5) + NUM_TEMPLATE_TYPES
    // = 3*16 + 5 + 4 = 57
    size_t raw_dim = LanguageConfig::TOP_K_CONCEPTS * LanguageConfig::ENCODER_QUERY_DIM
                     + 5 + LanguageConfig::NUM_TEMPLATE_TYPES;

    double limit = std::sqrt(6.0 / (raw_dim + LanguageConfig::FUSED_DIM));
    std::uniform_real_distribution<double> udist(-limit, limit);

    projection_.resize(raw_dim);
    for (auto& row : projection_) {
        row.resize(LanguageConfig::FUSED_DIM);
        for (auto& v : row) v = udist(rng);
    }
}

// =============================================================================
// Fusion
// =============================================================================

FusedRepresentation FusionLayer::fuse(
    const std::unordered_map<ConceptId, std::vector<double>>& activations,
    const SemanticScores& scores,
    const std::vector<ConceptId>& causal_chain
) const {
    FusedRepresentation result;
    result.template_type = scores.best_template();

    // Compute gate score for each active concept
    std::unordered_map<ConceptId, double> gates;
    std::vector<std::pair<ConceptId, double>> scored;

    for (const auto& [cid, act] : activations) {
        double norm = vec_norm(act);
        double rel = 0.0;
        auto rel_it = scores.relevance.find(cid);
        if (rel_it != scores.relevance.end()) rel = rel_it->second;

        // Find max causality involving this concept
        double caus = 0.0;
        for (const auto& [key, val] : scores.causality) {
            size_t colon = key.find(':');
            if (colon != std::string::npos) {
                ConceptId src = std::stoull(key.substr(0, colon));
                ConceptId tgt = std::stoull(key.substr(colon + 1));
                if (src == cid || tgt == cid) {
                    caus = std::max(caus, val);
                }
            }
        }

        double gate = compute_gate(norm, rel, caus);
        gates[cid] = gate;
        result.gate_scores[cid] = gate;
        scored.push_back({cid, gate * norm});
    }

    // Order: first by causal chain position, then by score
    std::unordered_map<ConceptId, int> chain_pos;
    for (size_t i = 0; i < causal_chain.size(); ++i) {
        chain_pos[causal_chain[i]] = static_cast<int>(i);
    }

    std::sort(scored.begin(), scored.end(),
        [&](const auto& a, const auto& b) {
            bool a_in_chain = chain_pos.count(a.first) > 0;
            bool b_in_chain = chain_pos.count(b.first) > 0;
            if (a_in_chain && !b_in_chain) return true;
            if (!a_in_chain && b_in_chain) return false;
            if (a_in_chain && b_in_chain) {
                return chain_pos[a.first] < chain_pos[b.first];
            }
            return a.second > b.second;
        });

    for (const auto& [cid, score] : scored) {
        result.ordered_concepts.push_back(cid);
    }

    // Build raw fused vector and project to FUSED_DIM
    auto raw = build_raw_fused(scored, activations, gates, scores.template_probs);

    // Project: raw × projection → fused (matrix multiply)
    result.fused_vector.resize(LanguageConfig::FUSED_DIM, 0.0);
    size_t raw_dim = std::min(raw.size(), projection_.size());
    for (size_t i = 0; i < raw_dim; ++i) {
        for (size_t j = 0; j < LanguageConfig::FUSED_DIM; ++j) {
            result.fused_vector[j] += raw[i] * projection_[i][j];
        }
    }

    return result;
}

// =============================================================================
// Helpers
// =============================================================================

double FusionLayer::compute_gate(double activation_norm, double relevance, double causality) const {
    double z = gate_b_;
    std::vector<double> input = {activation_norm, relevance, causality};
    for (size_t i = 0; i < std::min(input.size(), gate_w_.size()); ++i) {
        z += gate_w_[i] * input[i];
    }
    return sigmoid(z);
}

std::vector<double> FusionLayer::build_raw_fused(
    const std::vector<std::pair<ConceptId, double>>& top_concepts,
    const std::unordered_map<ConceptId, std::vector<double>>& activations,
    const std::unordered_map<ConceptId, double>& gates,
    const std::vector<double>& template_probs
) const {
    // Raw: [top3 × 16D activations | 5 gate scores | 4 template probs]
    const size_t act_dim = LanguageConfig::ENCODER_QUERY_DIM;
    std::vector<double> raw;
    raw.reserve(LanguageConfig::TOP_K_CONCEPTS * act_dim + 5 + LanguageConfig::NUM_TEMPLATE_TYPES);

    // Top-K concept activations (weighted by gate)
    for (size_t k = 0; k < LanguageConfig::TOP_K_CONCEPTS; ++k) {
        if (k < top_concepts.size()) {
            auto act_it = activations.find(top_concepts[k].first);
            auto gate_it = gates.find(top_concepts[k].first);
            double g = (gate_it != gates.end()) ? gate_it->second : 0.5;

            if (act_it != activations.end()) {
                for (size_t d = 0; d < act_dim; ++d) {
                    double val = (d < act_it->second.size()) ? act_it->second[d] : 0.0;
                    raw.push_back(val * g);
                }
            } else {
                for (size_t d = 0; d < act_dim; ++d) raw.push_back(0.0);
            }
        } else {
            for (size_t d = 0; d < act_dim; ++d) raw.push_back(0.0);
        }
    }

    // Top-5 gate scores (or pad with 0)
    for (size_t k = 0; k < 5; ++k) {
        if (k < top_concepts.size()) {
            auto gate_it = gates.find(top_concepts[k].first);
            raw.push_back(gate_it != gates.end() ? gate_it->second : 0.0);
        } else {
            raw.push_back(0.0);
        }
    }

    // Template type probabilities
    for (size_t t = 0; t < LanguageConfig::NUM_TEMPLATE_TYPES; ++t) {
        raw.push_back(t < template_probs.size() ? template_probs[t] : 0.0);
    }

    return raw;
}

// =============================================================================
// Extended Fused Vector
// =============================================================================

std::vector<double> FusedRepresentation::extended_fused_vector() const {
    std::vector<double> result;
    result.reserve(fused_vector.size() + dimensional_context.size());
    result = fused_vector;
    result.insert(result.end(), dimensional_context.begin(), dimensional_context.end());
    return result;
}

// =============================================================================
// Helpers
// =============================================================================

double FusionLayer::sigmoid(double x) {
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

double FusionLayer::vec_norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

} // namespace brain19
