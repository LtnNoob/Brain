#include "relevance_map.hpp"

#include <cmath>
#include <limits>
#include <numeric>

namespace brain19 {

// =============================================================================
// Compute
// =============================================================================

RelevanceMap RelevanceMap::compute(
        ConceptId source,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm,
        RelationType rel_type,
        const std::string& context) {

    RelevanceMap map(source);

    MicroModel* model = registry.get_model(source);
    if (!model) return map;

    const Vec10& e = embeddings.get_relation_embedding(rel_type);

    // Pre-compute context hash once (avoids per-target string allocation)
    size_t context_hash = std::hash<std::string>{}(context);

    auto all_ids = ltm.get_all_concept_ids();
    for (ConceptId cid : all_ids) {
        if (cid == source) continue;  // Skip self
        // Performance fix: numeric target embedding without string allocation
        Vec10 c = embeddings.make_target_embedding(context_hash, source, cid);
        double score = model->predict(e, c);
        map.scores_[cid] = score;
    }

    return map;
}

// =============================================================================
// Query
// =============================================================================

double RelevanceMap::score(ConceptId cid) const {
    auto it = scores_.find(cid);
    if (it == scores_.end()) return 0.0;
    return it->second;
}

std::vector<std::pair<ConceptId, double>> RelevanceMap::top_k(size_t k) const {
    std::vector<std::pair<ConceptId, double>> entries(scores_.begin(), scores_.end());

    // Partial sort for efficiency
    if (k < entries.size()) {
        std::partial_sort(entries.begin(), entries.begin() + k, entries.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        entries.resize(k);
    } else {
        std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    return entries;
}

std::vector<std::pair<ConceptId, double>> RelevanceMap::above_threshold(double threshold) const {
    std::vector<std::pair<ConceptId, double>> result;
    for (const auto& [cid, s] : scores_) {
        if (s >= threshold) {
            result.push_back({cid, s});
        }
    }
    // Sort by score descending
    std::sort(result.begin(), result.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    return result;
}

// =============================================================================
// Overlay operations
// =============================================================================

void RelevanceMap::overlay(const RelevanceMap& other, OverlayMode mode, double weight) {
    for (const auto& [cid, s] : other.scores_) {
        auto it = scores_.find(cid);
        if (it == scores_.end()) {
            // New entry: use other's score (weighted if applicable)
            switch (mode) {
                case OverlayMode::ADDITION:
                    scores_[cid] = s;
                    break;
                case OverlayMode::MAX:
                    scores_[cid] = s;
                    break;
                case OverlayMode::WEIGHTED_AVERAGE:
                    scores_[cid] = s * weight;
                    break;
            }
        } else {
            switch (mode) {
                case OverlayMode::ADDITION:
                    it->second += s;
                    break;
                case OverlayMode::MAX:
                    it->second = std::max(it->second, s);
                    break;
                case OverlayMode::WEIGHTED_AVERAGE:
                    it->second = it->second * (1.0 - weight) + s * weight;
                    break;
            }
        }
    }
}

RelevanceMap RelevanceMap::combine(
        const std::vector<RelevanceMap>& maps,
        OverlayMode mode,
        const std::vector<double>& weights) {

    if (maps.empty()) return RelevanceMap(0);

    RelevanceMap result(maps[0].source_cid_);
    result.scores_ = maps[0].scores_;

    // Apply weights[0] to the copied base map
    double w0 = (0 < weights.size()) ? weights[0] : (1.0 / static_cast<double>(maps.size()));
    for (auto& [cid, s] : result.scores_) {
        s *= w0;
    }

    for (size_t i = 1; i < maps.size(); ++i) {
        double w = (i < weights.size()) ? weights[i] : (1.0 / static_cast<double>(maps.size()));
        result.overlay(maps[i], mode, w);
    }

    return result;
}

// =============================================================================
// Normalize
// =============================================================================

void RelevanceMap::normalize() {
    if (scores_.empty()) return;

    double max_val = -std::numeric_limits<double>::infinity();
    double min_val = std::numeric_limits<double>::infinity();

    for (const auto& [cid, s] : scores_) {
        max_val = std::max(max_val, s);
        min_val = std::min(min_val, s);
    }

    double range = max_val - min_val;
    if (range < 1e-12) {
        // All scores are the same; set to 0.5
        for (auto& [cid, s] : scores_) {
            s = 0.5;
        }
        return;
    }

    for (auto& [cid, s] : scores_) {
        s = (s - min_val) / range;
    }
}

} // namespace brain19
