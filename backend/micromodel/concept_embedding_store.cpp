#include "concept_embedding_store.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace brain19 {

FlexEmbedding ConceptEmbeddingStore::hash_init(ConceptId cid) {
    FlexEmbedding emb;
    // SplitMix64-inspired hash for deterministic initialization
    uint64_t x = static_cast<uint64_t>(cid);
    for (size_t i = 0; i < CORE_DIM; ++i) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        // Map to [-0.5, 0.5]
        emb.core[i] = (static_cast<double>(z) / static_cast<double>(UINT64_MAX)) - 0.5;
    }

    // Normalize core to unit length
    double norm = 0.0;
    for (double v : emb.core) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-10) {
        for (double& v : emb.core) v /= norm;
    }
    // detail starts empty (0 additional dimensions)
    return emb;
}

const FlexEmbedding& ConceptEmbeddingStore::get(ConceptId cid) {
    auto it = store_.find(cid);
    if (it != store_.end()) return it->second;
    auto [ins, _] = store_.emplace(cid, hash_init(cid));
    return ins->second;
}

void ConceptEmbeddingStore::set(ConceptId cid, const FlexEmbedding& emb) {
    store_[cid] = emb;
}

void ConceptEmbeddingStore::nudge(ConceptId cid, const FlexEmbedding& target, double alpha) {
    auto it = store_.find(cid);
    if (it == store_.end()) {
        auto [ins, ok] = store_.emplace(cid, hash_init(cid));
        it = ins;
    }
    auto& emb = it->second;
    // Always nudge core
    for (size_t i = 0; i < CORE_DIM; ++i) {
        emb.core[i] = (1.0 - alpha) * emb.core[i] + alpha * target.core[i];
    }
    // Nudge shared detail dimensions
    size_t shared_detail = std::min(emb.detail.size(), target.detail.size());
    for (size_t i = 0; i < shared_detail; ++i) {
        emb.detail[i] = (1.0 - alpha) * emb.detail[i] + alpha * target.detail[i];
    }
}

double ConceptEmbeddingStore::similarity(ConceptId a, ConceptId b) {
    // Copy by value: get(b) may rehash, invalidating a reference from get(a)
    const FlexEmbedding ea = get(a);
    const FlexEmbedding eb = get(b);
    return full_similarity(ea, eb);
}

std::vector<std::pair<ConceptId, double>> ConceptEmbeddingStore::most_similar(ConceptId cid, size_t k) {
    // Ensure the query concept has an embedding
    const FlexEmbedding query = get(cid);

    // Phase 1: Core-similarity for all concepts -> top 50 candidates
    static constexpr size_t CORE_FILTER_K = 50;
    std::vector<std::pair<ConceptId, double>> candidates;
    candidates.reserve(store_.size());

    for (const auto& [other_cid, other_emb] : store_) {
        if (other_cid == cid) continue;
        double cs = core_similarity(query, other_emb);
        candidates.emplace_back(other_cid, cs);
    }

    // Get top CORE_FILTER_K by core similarity
    size_t filter_k = std::min(CORE_FILTER_K, candidates.size());
    if (filter_k < candidates.size()) {
        std::partial_sort(candidates.begin(), candidates.begin() + filter_k, candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        candidates.resize(filter_k);
    }

    // Phase 2: Full similarity for top candidates -> rerank
    for (auto& [other_cid, score] : candidates) {
        auto it = store_.find(other_cid);
        if (it != store_.end()) {
            score = full_similarity(query, it->second);
        }
    }

    // Sort by full similarity and take top K
    if (k < candidates.size()) {
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        candidates.resize(k);
    } else {
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }
    return candidates;
}

bool ConceptEmbeddingStore::has(ConceptId cid) const {
    return store_.count(cid) > 0;
}

} // namespace brain19
