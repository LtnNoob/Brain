#include "knowledge_extractor.hpp"

#include <algorithm>
#include <cmath>

namespace brain19 {

KnowledgeExtractor::KnowledgeExtractor(
    LongTermMemory& ltm, GraphReasoner& reasoner,
    const CoLearnConfig& config)
    : ltm_(ltm)
    , reasoner_(reasoner)
    , config_(config)
{
}

RelationId KnowledgeExtractor::find_relation(
    ConceptId source, ConceptId target, RelationType type) const
{
    auto rels = ltm_.get_relations_between(source, target);
    for (const auto& rel : rels) {
        if (rel.type == type) return rel.id;
    }
    return 0;
}

ConsolidationResult KnowledgeExtractor::consolidate_episode(const Episode& episode) {
    ConsolidationResult result;

    if (episode.steps.size() < 2) return result;

    // Re-reason from the episode's seed
    GraphChain new_chain = reasoner_.reason_from(episode.seed);
    double new_quality = reasoner_.compute_chain_quality(new_chain);

    // Extract signals from the new chain
    ChainSignal signal = reasoner_.extract_signals(new_chain);

    // Quality differential: positive means we improved
    double quality_delta = new_quality - episode.quality;

    // Apply signals with quality-modulated strength
    for (const auto& suggestion : signal.suggestions) {
        RelationId rid = find_relation(suggestion.source, suggestion.target, suggestion.relation);
        if (rid == 0) continue;

        auto rel = ltm_.get_relation(rid);
        if (!rel) continue;

        double delta = suggestion.delta_weight * config_.weight_delta;

        // Cycle-based LR decay: prevents oscillation, ensures convergence
        delta *= 1.0 / (1.0 + config_.lr_decay_rate * static_cast<double>(current_cycle_));

        // Modulate by quality delta (symmetric):
        //   Quality improved → trust signals more (boost both directions)
        //   Quality degraded → trust signals less (dampen both directions)
        double confidence = 1.0 + std::clamp(quality_delta, -0.5, 0.5);
        delta *= confidence;

        double new_weight = rel->weight + delta;

        double abs_delta = std::abs(delta);
        if (new_weight < config_.prune_threshold) {
            // Prune: remove the relation
            ltm_.remove_relation(rid);
            ++result.edges_pruned;
            cumulative_changes_[suggestion.source] += abs_delta;
            cumulative_changes_[suggestion.target] += abs_delta;
        } else {
            ltm_.modify_relation_weight(rid, new_weight);
            if (delta > 0.0) {
                ++result.edges_strengthened;
            } else if (delta < 0.0) {
                ++result.edges_weakened;
            }
            cumulative_changes_[suggestion.source] += abs_delta;
            cumulative_changes_[suggestion.target] += abs_delta;
        }
    }

    ++result.episodes_consolidated;
    return result;
}

ConsolidationResult KnowledgeExtractor::consolidate_batch(
    const std::vector<const Episode*>& episodes)
{
    ConsolidationResult total;
    for (const auto* ep : episodes) {
        if (!ep) continue;
        total += consolidate_episode(*ep);
    }
    total.episodes_replayed = episodes.size();
    return total;
}

size_t KnowledgeExtractor::apply_signals(const ChainSignal& signal) {
    size_t applied = 0;

    for (const auto& suggestion : signal.suggestions) {
        RelationId rid = find_relation(suggestion.source, suggestion.target, suggestion.relation);
        if (rid == 0) continue;

        auto rel = ltm_.get_relation(rid);
        if (!rel) continue;

        double new_weight = rel->weight + suggestion.delta_weight;

        if (new_weight < config_.prune_threshold) {
            ltm_.remove_relation(rid);
        } else {
            ltm_.modify_relation_weight(rid, new_weight);
        }
        ++applied;
    }

    return applied;
}

} // namespace brain19
