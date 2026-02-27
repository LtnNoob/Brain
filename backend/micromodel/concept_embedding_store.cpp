#include "concept_embedding_store.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../core/relation_config.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>

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

    // Initialize detail dimensions (INITIAL_DETAIL = 16)
    // Uses continued hash sequence for concept-specific detail features.
    // These provide the cyclic_compress input and dim_fraction signal
    // that ConceptModel's MultiHeadBilinear and FlexKAN depend on.
    emb.detail.resize(FlexConfig::INITIAL_DETAIL);
    for (size_t i = 0; i < FlexConfig::INITIAL_DETAIL; ++i) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        // Smaller scale for detail dims: [-0.25, 0.25]
        emb.detail[i] = (static_cast<double>(z) / static_cast<double>(UINT64_MAX)) - 0.5;
        emb.detail[i] *= 0.5;
    }

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
    // Nudge existing shared detail dimensions first
    size_t old_size = emb.detail.size();
    size_t shared_detail = std::min(old_size, target.detail.size());
    for (size_t i = 0; i < shared_detail; ++i) {
        emb.detail[i] = (1.0 - alpha) * emb.detail[i] + alpha * target.detail[i];
    }
    // Expand detail dims if target has more (grow toward richer representations)
    if (target.detail.size() > old_size) {
        emb.detail.resize(target.detail.size(), 0.0);
        // New dims blend from zero toward target
        for (size_t i = old_size; i < emb.detail.size(); ++i) {
            emb.detail[i] = alpha * target.detail[i];
        }
    }

    // Re-normalize core to unit length (preserve cosine similarity invariant)
    double norm_sq = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        norm_sq += emb.core[i] * emb.core[i];
    }
    double norm = std::sqrt(norm_sq);
    if (norm > 1e-10) {
        double inv_norm = 1.0 / norm;
        for (size_t i = 0; i < CORE_DIM; ++i) {
            emb.core[i] *= inv_norm;
        }
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

FlexEmbedding ConceptEmbeddingStore::get_or_default(ConceptId cid) const {
    auto it = store_.find(cid);
    if (it != store_.end()) return it->second;
    return hash_init(cid);
}

bool ConceptEmbeddingStore::has(ConceptId cid) const {
    return store_.count(cid) > 0;
}

// =============================================================================
// Learn from graph: nudge embeddings toward neighbor averages
// =============================================================================
//
// For each concept C with relations in LTM:
//   1. Collect all neighbor embeddings (outgoing targets + incoming sources)
//   2. Weight each neighbor by its relation weight
//   3. Compute weighted average embedding
//   4. Nudge C toward this average with alpha
//
// After multiple iterations, connected concepts converge in embedding space.
// IS_A neighbors, HAS_PROPERTY targets, CAUSES targets etc. all pull
// the concept's embedding toward a semantically meaningful region.

ConceptEmbeddingStore::LearnResult
ConceptEmbeddingStore::learn_from_graph(const LongTermMemory& ltm,
                                         double alpha, size_t iterations) {
    LearnConfig config;
    config.alpha = alpha;
    config.iterations = iterations;
    return learn_from_graph(ltm, config);
}

ConceptEmbeddingStore::LearnResult
ConceptEmbeddingStore::learn_from_graph(const LongTermMemory& ltm,
                                         const LearnConfig& config) {
    LearnResult result;
    auto all_ids = ltm.get_all_concept_ids();

    // Ensure all concepts have embeddings
    for (auto cid : all_ids) {
        get(cid);
    }

    std::mt19937 rng(config.rng_seed);
    std::uniform_int_distribution<size_t> dist(0, all_ids.empty() ? 0 : all_ids.size() - 1);
    double neg_alpha = config.alpha * config.negative_alpha_ratio;

    for (size_t iter = 0; iter < config.iterations; ++iter) {
        result.iterations = iter + 1;

        for (auto cid : all_ids) {
            // Skip anti-knowledge concepts (they don't need embedding updates)
            auto cid_info = ltm.retrieve_concept(cid);
            if (cid_info && cid_info->is_anti_knowledge) continue;

            // Build weighted average of neighbor embeddings
            FlexEmbedding avg;
            avg.detail.resize(FlexConfig::INITIAL_DETAIL, 0.0);
            double total_weight = 0.0;
            size_t neighbor_count = 0;
            std::unordered_set<ConceptId> neighbor_ids;

            auto outgoing = ltm.get_outgoing_relations(cid);
            for (const auto& rel : outgoing) {
                auto neighbor_info = ltm.retrieve_concept(rel.target);

                // Anti-Knowledge: simple AK = skip, complex AK = repulsive nudge
                if (neighbor_info && neighbor_info->is_anti_knowledge
                    && neighbor_info->complexity_score < 0.3f) continue;
                bool is_ak = neighbor_info && neighbor_info->is_anti_knowledge;

                neighbor_ids.insert(rel.target);
                const auto& neighbor_emb = get(rel.target);

                // Relation-type-aware alpha (Convergence v2, Audit #2)
                const RelationBehavior& behavior = get_behavior(rel.type);
                double w = rel.weight * behavior.embedding_alpha;
                // w is NEGATIVE for OPPOSITION → repulsive nudge
                // Complex AK: also repulsive (negate)
                if (is_ak) w = -std::abs(w);

                // Weight by neighbor's epistemic trust
                if (neighbor_info) w *= neighbor_info->epistemic.trust;

                for (size_t i = 0; i < CORE_DIM; ++i) {
                    avg.core[i] += w * neighbor_emb.core[i];
                }
                size_t shared = std::min(avg.detail.size(), neighbor_emb.detail.size());
                for (size_t i = 0; i < shared; ++i) {
                    avg.detail[i] += w * neighbor_emb.detail[i];
                }
                total_weight += std::abs(w);
                ++neighbor_count;
            }

            auto incoming = ltm.get_incoming_relations(cid);
            for (const auto& rel : incoming) {
                auto neighbor_info = ltm.retrieve_concept(rel.source);

                // Anti-Knowledge: simple AK = skip, complex AK = repulsive nudge
                if (neighbor_info && neighbor_info->is_anti_knowledge
                    && neighbor_info->complexity_score < 0.3f) continue;
                bool is_ak = neighbor_info && neighbor_info->is_anti_knowledge;

                neighbor_ids.insert(rel.source);
                const auto& neighbor_emb = get(rel.source);

                // Relation-type-aware alpha for incoming (weaker influence)
                const RelationBehavior& behavior = get_behavior(rel.type);
                double w = rel.weight * behavior.embedding_alpha * 0.5;
                if (is_ak) w = -std::abs(w);

                // Weight by neighbor's epistemic trust
                if (neighbor_info) w *= neighbor_info->epistemic.trust;

                for (size_t i = 0; i < CORE_DIM; ++i) {
                    avg.core[i] += w * neighbor_emb.core[i];
                }
                size_t shared = std::min(avg.detail.size(), neighbor_emb.detail.size());
                for (size_t i = 0; i < shared; ++i) {
                    avg.detail[i] += w * neighbor_emb.detail[i];
                }
                total_weight += std::abs(w);
                ++neighbor_count;
            }

            if (total_weight < 1e-10) continue;

            // Normalize
            double inv_w = 1.0 / total_weight;
            for (size_t i = 0; i < CORE_DIM; ++i) {
                avg.core[i] *= inv_w;
            }
            for (size_t i = 0; i < avg.detail.size(); ++i) {
                avg.detail[i] *= inv_w;
            }

            // Nudge toward average (positive)
            nudge(cid, avg, config.alpha);
            result.concepts_updated++;
            result.total_neighbors += neighbor_count;

            // --- Contrastive negative sampling ---
            if (config.negative_samples > 0 && all_ids.size() > neighbor_ids.size() + 1) {
                FlexEmbedding neg_avg;
                neg_avg.detail.resize(FlexConfig::INITIAL_DETAIL, 0.0);
                size_t sampled = 0;
                size_t max_attempts = config.negative_samples * 3;

                for (size_t attempt = 0; attempt < max_attempts && sampled < config.negative_samples; ++attempt) {
                    ConceptId neg_cid = all_ids[dist(rng)];
                    if (neg_cid == cid || neighbor_ids.count(neg_cid)) continue;

                    const auto& neg_emb = get(neg_cid);
                    for (size_t i = 0; i < CORE_DIM; ++i) {
                        neg_avg.core[i] += neg_emb.core[i];
                    }
                    size_t shared = std::min(neg_avg.detail.size(), neg_emb.detail.size());
                    for (size_t i = 0; i < shared; ++i) {
                        neg_avg.detail[i] += neg_emb.detail[i];
                    }
                    ++sampled;
                }

                if (sampled > 0) {
                    double inv_s = 1.0 / static_cast<double>(sampled);
                    const auto& current = get(cid);

                    // anti_target = mirror of neg_avg through current embedding
                    FlexEmbedding anti;
                    anti.detail.resize(FlexConfig::INITIAL_DETAIL, 0.0);
                    for (size_t i = 0; i < CORE_DIM; ++i) {
                        anti.core[i] = 2.0 * current.core[i] - neg_avg.core[i] * inv_s;
                    }
                    size_t shared = std::min(anti.detail.size(), current.detail.size());
                    for (size_t i = 0; i < shared; ++i) {
                        anti.detail[i] = 2.0 * current.detail[i] - neg_avg.detail[i] * inv_s;
                    }

                    nudge(cid, anti, neg_alpha);
                }
            }
        }
    }

    return result;
}

} // namespace brain19
