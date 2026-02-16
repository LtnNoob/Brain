#pragma once

#include "convergence_config.hpp"
#include "../memory/relation_type_registry.hpp"
#include <vector>
#include <unordered_map>
#include <cassert>

namespace brain19 {
namespace convergence {

// =============================================================================
// RELATION-AWARE DECODER — Interprets convergence output by relation category
// =============================================================================
//
// The 32-dim convergence output is allocated by RelationCategory:
//   HIERARCHICAL:  dims 0-3   (4 dims: IS_A, INSTANCE_OF)
//   COMPOSITIONAL: dims 4-7   (4 dims: PART_OF, CONTAINS)
//   CAUSAL:        dims 8-11  (4 dims: CAUSES, ENABLES)
//   OPPOSITION:    dims 12-13 (2 dims: CONTRADICTS)
//   SIMILARITY:    dims 14-19 (6 dims: SIMILAR_TO, ASSOCIATED_WITH)
//   TEMPORAL:      dims 20-23 (4 dims: BEFORE, AFTER)
//   FUNCTIONAL:    dims 24-27 (4 dims: REQUIRES, USES)
//   EPISTEMIC:     dims 28-29 (2 dims: SUPPORTS)
//   CUSTOM:        dims 30-31 (2 dims: reserved)
//

struct CategorySlice {
    size_t start;
    size_t end;  // exclusive
    size_t size() const { return end - start; }
};

class RelationDecoder {
public:
    RelationDecoder();

    // Get the slice for a category
    const CategorySlice& get_slice(RelationCategory cat) const;

    // Extract the sub-vector for a category from full output
    std::vector<double> decode(const std::vector<double>& logits,
                                RelationCategory cat) const;

    // Extract scalar prediction: mean of category slice
    double decode_scalar(const std::vector<double>& logits,
                          RelationCategory cat) const;

    // Get the most active category in the output
    RelationCategory dominant_category(const std::vector<double>& logits) const;

    // Get category activations (mean absolute value per category)
    std::vector<std::pair<RelationCategory, double>> category_activations(
        const std::vector<double>& logits) const;

private:
    std::unordered_map<RelationCategory, CategorySlice> slices_;
    void init_slices();
};

// Inline implementation (small enough to be header-only)

inline RelationDecoder::RelationDecoder() {
    init_slices();
}

inline void RelationDecoder::init_slices() {
    slices_[RelationCategory::HIERARCHICAL]   = {0, 4};
    slices_[RelationCategory::COMPOSITIONAL]  = {4, 8};
    slices_[RelationCategory::CAUSAL]         = {8, 12};
    slices_[RelationCategory::OPPOSITION]     = {12, 14};
    slices_[RelationCategory::SIMILARITY]     = {14, 20};
    slices_[RelationCategory::TEMPORAL]       = {20, 24};
    slices_[RelationCategory::FUNCTIONAL]     = {24, 28};
    slices_[RelationCategory::EPISTEMIC]      = {28, 30};
    slices_[RelationCategory::CUSTOM_CATEGORY]= {30, 32};
}

inline const CategorySlice& RelationDecoder::get_slice(RelationCategory cat) const {
    static const CategorySlice EMPTY{0, 0};
    auto it = slices_.find(cat);
    if (it != slices_.end()) return it->second;
    return EMPTY;
}

inline std::vector<double> RelationDecoder::decode(
    const std::vector<double>& logits, RelationCategory cat) const
{
    assert(logits.size() >= OUTPUT_DIM);
    const auto& s = get_slice(cat);
    return std::vector<double>(logits.begin() + s.start, logits.begin() + s.end);
}

inline double RelationDecoder::decode_scalar(
    const std::vector<double>& logits, RelationCategory cat) const
{
    const auto& s = get_slice(cat);
    if (s.size() == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = s.start; i < s.end && i < logits.size(); ++i) {
        sum += logits[i];
    }
    return sum / static_cast<double>(s.size());
}

inline RelationCategory RelationDecoder::dominant_category(
    const std::vector<double>& logits) const
{
    RelationCategory best = RelationCategory::HIERARCHICAL;
    double best_score = -1e30;
    for (const auto& [cat, slice] : slices_) {
        double score = 0.0;
        for (size_t i = slice.start; i < slice.end && i < logits.size(); ++i) {
            score += std::abs(logits[i]);
        }
        score /= static_cast<double>(slice.size());
        if (score > best_score) {
            best_score = score;
            best = cat;
        }
    }
    return best;
}

inline std::vector<std::pair<RelationCategory, double>>
RelationDecoder::category_activations(const std::vector<double>& logits) const {
    std::vector<std::pair<RelationCategory, double>> result;
    for (const auto& [cat, slice] : slices_) {
        double score = 0.0;
        for (size_t i = slice.start; i < slice.end && i < logits.size(); ++i) {
            score += std::abs(logits[i]);
        }
        result.emplace_back(cat, score / static_cast<double>(slice.size()));
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return result;
}

} // namespace convergence
} // namespace brain19
