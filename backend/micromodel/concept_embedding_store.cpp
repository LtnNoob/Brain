#include "concept_embedding_store.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace brain19 {

Vec10 ConceptEmbeddingStore::hash_init(ConceptId cid) {
    Vec10 emb{};
    // SplitMix64-inspired hash for deterministic initialization
    uint64_t x = static_cast<uint64_t>(cid);
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        // Map to [-0.5, 0.5]
        emb[i] = (static_cast<double>(z) / static_cast<double>(UINT64_MAX)) - 0.5;
    }

    // Normalize to unit length
    double norm = 0.0;
    for (double v : emb) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-10) {
        for (double& v : emb) v /= norm;
    }
    return emb;
}

const Vec10& ConceptEmbeddingStore::get(ConceptId cid) {
    auto it = store_.find(cid);
    if (it != store_.end()) return it->second;
    auto [ins, _] = store_.emplace(cid, hash_init(cid));
    return ins->second;
}

void ConceptEmbeddingStore::set(ConceptId cid, const Vec10& emb) {
    store_[cid] = emb;
}

void ConceptEmbeddingStore::nudge(ConceptId cid, const Vec10& target, double alpha) {
    auto it = store_.find(cid);
    if (it == store_.end()) {
        auto [ins, ok] = store_.emplace(cid, hash_init(cid));
        it = ins;
    }
    auto& emb = it->second;
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        emb[i] = (1.0 - alpha) * emb[i] + alpha * target[i];
    }
}

double ConceptEmbeddingStore::similarity(ConceptId a, ConceptId b) {
    // Copy by value: get(b) may rehash, invalidating a reference from get(a)
    const Vec10 ea = get(a);
    const Vec10 eb = get(b);

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        dot += ea[i] * eb[i];
        na += ea[i] * ea[i];
        nb += eb[i] * eb[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 1e-10) ? (dot / denom) : 0.0;
}

std::vector<std::pair<ConceptId, double>> ConceptEmbeddingStore::most_similar(ConceptId cid, size_t k) {
    // Ensure the query concept has an embedding
    get(cid);

    std::vector<std::pair<ConceptId, double>> results;
    results.reserve(store_.size());

    for (const auto& [other_cid, _] : store_) {
        if (other_cid == cid) continue;
        results.emplace_back(other_cid, similarity(cid, other_cid));
    }

    // Partial sort for top-k
    if (k < results.size()) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }
    return results;
}

bool ConceptEmbeddingStore::has(ConceptId cid) const {
    return store_.count(cid) > 0;
}

} // namespace brain19
