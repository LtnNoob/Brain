#include "dimensional_context.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace brain19 {

// Static empty profile for missing concepts
const DimensionalContext::DimProfile DimensionalContext::empty_profile_ = {};

// =============================================================================
// Build — discover dimensions from graph structure
// =============================================================================

void DimensionalContext::build(const LongTermMemory& ltm) {
    auto& reg = RelationTypeRegistry::instance();
    profiles_.clear();
    dim_order_.clear();
    dim_index_.clear();
    max_dimensionality_ = 0;

    // Track all categories observed across the entire graph
    std::set<RelationCategory> all_observed;

    for (auto cid : ltm.get_all_concept_ids()) {
        DimProfile profile;

        // Outgoing relations: full weight (1.0 per relation)
        auto out_rels = ltm.get_outgoing_relations(cid);
        for (const auto& r : out_rels) {
            auto cat = reg.get_category(r.type);
            if (cat == RelationCategory::CUSTOM_CATEGORY) continue;
            profile.weights[cat] += 1.0;
            all_observed.insert(cat);
        }

        // Incoming relations: half weight (directed graph asymmetry)
        auto in_rels = ltm.get_incoming_relations(cid);
        for (const auto& r : in_rels) {
            auto cat = reg.get_category(r.type);
            if (cat == RelationCategory::CUSTOM_CATEGORY) continue;
            profile.weights[cat] += 0.5;
            all_observed.insert(cat);
        }

        // L1-normalize (distribution sums to 1.0)
        double sum = 0.0;
        for (const auto& [cat, w] : profile.weights) sum += w;
        if (sum > 1e-12) {
            for (auto& [cat, w] : profile.weights) w /= sum;
        }

        if (profile.dimensionality() > max_dimensionality_) {
            max_dimensionality_ = profile.dimensionality();
        }

        profiles_[cid] = std::move(profile);
    }

    // Build ordered dimension index from observed categories.
    // Sort by enum value for deterministic ordering.
    dim_order_.assign(all_observed.begin(), all_observed.end());
    std::sort(dim_order_.begin(), dim_order_.end(),
        [](RelationCategory a, RelationCategory b) {
            return static_cast<uint8_t>(a) < static_cast<uint8_t>(b);
        });

    for (size_t i = 0; i < dim_order_.size(); ++i) {
        dim_index_[dim_order_[i]] = i;
    }

    built_ = true;
}

// =============================================================================
// Profile lookup
// =============================================================================

const DimensionalContext::DimProfile& DimensionalContext::get_profile(ConceptId cid) const {
    auto it = profiles_.find(cid);
    return (it != profiles_.end()) ? it->second : empty_profile_;
}

// =============================================================================
// Decoder vector projection
// =============================================================================

std::vector<double> DimensionalContext::to_decoder_vec(ConceptId cid) const {
    const size_t od = dim_order_.size();
    std::vector<double> vec(od + 2, 0.0);

    auto it = profiles_.find(cid);
    if (it == profiles_.end()) return vec;

    const auto& profile = it->second;

    // Sparse weights expanded to observed-dimension slots
    for (const auto& [cat, weight] : profile.weights) {
        auto idx_it = dim_index_.find(cat);
        if (idx_it != dim_index_.end()) {
            vec[idx_it->second] = weight;
        }
    }

    // Richness: this concept's dimensionality / max observed in graph
    vec[od] = (max_dimensionality_ > 0)
        ? static_cast<double>(profile.dimensionality()) / static_cast<double>(max_dimensionality_)
        : 0.0;

    // Normalized Shannon entropy of the weight distribution
    double entropy = 0.0;
    for (const auto& [cat, w] : profile.weights) {
        if (w > 1e-12) entropy -= w * std::log(w);
    }
    double max_entropy = (profile.dimensionality() > 1)
        ? std::log(static_cast<double>(profile.dimensionality()))
        : 1.0;
    vec[od + 1] = (max_entropy > 1e-12) ? entropy / max_entropy : 0.0;

    return vec;
}

std::vector<double> DimensionalContext::average_decoder_vec(const std::vector<ConceptId>& cids) const {
    const size_t dim = decoder_dim();
    std::vector<double> avg(dim, 0.0);

    if (cids.empty()) return avg;

    for (auto cid : cids) {
        auto v = to_decoder_vec(cid);
        for (size_t i = 0; i < dim && i < v.size(); ++i) {
            avg[i] += v[i];
        }
    }

    double inv_n = 1.0 / static_cast<double>(cids.size());
    for (auto& x : avg) x *= inv_n;

    return avg;
}

} // namespace brain19
