#include "episodic_memory.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

namespace brain19 {

EpisodicMemory::EpisodicMemory(size_t max_episodes)
    : max_episodes_(max_episodes)
{
}

uint64_t EpisodicMemory::store(const Episode& episode) {
    // Evict if at capacity
    if (episodes_.size() >= max_episodes_) {
        evict_consolidated(max_episodes_ - 1);
    }

    uint64_t id = next_id_++;
    Episode ep = episode;
    ep.id = id;

    // Set timestamp if not already set
    if (ep.timestamp_ms == 0) {
        auto now = std::chrono::steady_clock::now();
        ep.timestamp_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count());
    }

    // Build concept index
    for (const auto& step : ep.steps) {
        concept_index_[step.concept_id].push_back(id);
    }

    episodes_.emplace(id, std::move(ep));
    return id;
}

Episode EpisodicMemory::from_chain(const GraphChain& chain, ConceptId seed) const {
    Episode ep;
    ep.seed = seed;
    ep.quality = 0.0;
    ep.termination = chain.termination;

    // Compute quality from chain metrics
    if (chain.steps.size() >= 2) {
        ep.quality = chain.chain_trust *
            std::log(1.0 + static_cast<double>(chain.steps.size())) / 3.0;
    }

    // Convert each TraceStep to an EpisodeStep
    for (const auto& trace : chain.steps) {
        EpisodeStep step;
        step.concept_id = (trace.step_index == 0) ? trace.source_id : trace.target_id;
        step.relation = trace.relation;
        step.from_concept = (trace.step_index == 0) ? 0 : trace.source_id;
        step.activation = trace.output_activation.core();
        step.step_trust = trace.step_trust;
        step.nn_quality = trace.nn_quality;
        step.kan_quality = trace.kan_quality;
        step.kan_gate = trace.kan_gate;
        ep.steps.push_back(step);
    }

    return ep;
}

const Episode* EpisodicMemory::get(uint64_t id) const {
    auto it = episodes_.find(id);
    return it != episodes_.end() ? &it->second : nullptr;
}

std::vector<const Episode*> EpisodicMemory::select_for_replay(
    size_t count,
    double w_quality, double w_recency, double w_novelty) const
{
    if (episodes_.empty()) return {};

    // Find max timestamp for recency scoring
    uint64_t max_ts = 0;
    for (const auto& [id, ep] : episodes_) {
        if (ep.timestamp_ms > max_ts) max_ts = ep.timestamp_ms;
    }
    double ts_range = static_cast<double>(max_ts);
    if (ts_range < 1.0) ts_range = 1.0;

    // Score each episode
    struct Scored {
        const Episode* ep;
        double score;
    };
    std::vector<Scored> scored;
    scored.reserve(episodes_.size());

    for (const auto& [id, ep] : episodes_) {
        double quality_score = ep.quality;
        double recency_score = static_cast<double>(ep.timestamp_ms) / ts_range;
        // Novelty: inverse of replay count (less replayed = more novel)
        double novelty_score = 1.0 / (1.0 + static_cast<double>(ep.replay_count));

        double total = w_quality * quality_score
                     + w_recency * recency_score
                     + w_novelty * novelty_score;
        scored.push_back({&ep, total});
    }

    // Sort descending by score
    std::sort(scored.begin(), scored.end(),
        [](const Scored& a, const Scored& b) { return a.score > b.score; });

    // Take top-count
    std::vector<const Episode*> result;
    size_t n = std::min(count, scored.size());
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        result.push_back(scored[i].ep);
    }
    return result;
}

std::vector<const Episode*> EpisodicMemory::episodes_for_concept(ConceptId cid) const {
    std::vector<const Episode*> result;
    auto it = concept_index_.find(cid);
    if (it == concept_index_.end()) return result;

    for (uint64_t id : it->second) {
        auto ep_it = episodes_.find(id);
        if (ep_it != episodes_.end()) {
            result.push_back(&ep_it->second);
        }
    }
    return result;
}

void EpisodicMemory::mark_replayed(uint64_t id) {
    auto it = episodes_.find(id);
    if (it != episodes_.end()) {
        ++it->second.replay_count;
    }
}

void EpisodicMemory::mark_consolidated(uint64_t id, double strength) {
    auto it = episodes_.find(id);
    if (it != episodes_.end()) {
        it->second.consolidation_strength = std::min(1.0, std::max(0.0, strength));
    }
}

size_t EpisodicMemory::evict_consolidated(size_t target_count) {
    if (episodes_.size() <= target_count) return 0;

    // Collect episodes sorted by consolidation_strength (highest first = evict first)
    struct EvictCandidate {
        uint64_t id;
        double consolidation_strength;
    };
    std::vector<EvictCandidate> candidates;
    for (const auto& [id, ep] : episodes_) {
        if (ep.consolidation_strength > 0.5) {
            candidates.push_back({id, ep.consolidation_strength});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const EvictCandidate& a, const EvictCandidate& b) {
            return a.consolidation_strength > b.consolidation_strength;
        });

    size_t evicted = 0;
    for (const auto& c : candidates) {
        if (episodes_.size() <= target_count) break;

        // Remove from concept index
        auto ep_it = episodes_.find(c.id);
        if (ep_it != episodes_.end()) {
            for (const auto& step : ep_it->second.steps) {
                auto idx_it = concept_index_.find(step.concept_id);
                if (idx_it != concept_index_.end()) {
                    auto& vec = idx_it->second;
                    vec.erase(std::remove(vec.begin(), vec.end(), c.id), vec.end());
                    if (vec.empty()) concept_index_.erase(idx_it);
                }
            }
            episodes_.erase(ep_it);
            ++evicted;
        }
    }

    return evicted;
}

} // namespace brain19
