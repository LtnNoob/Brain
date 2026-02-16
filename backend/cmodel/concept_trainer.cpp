#include "concept_trainer.hpp"
#include "../core/relation_config.hpp"

#include <algorithm>
#include <thread>
#include <unordered_set>

namespace brain19 {

ConceptTrainer::ConceptTrainer(const ConceptTrainerConfig& config)
    : config_(config)
    , multihop_sampler_(config.multihop_config)
{}

// =============================================================================
// Sample generation — same logic as MicroTrainer
// =============================================================================

std::vector<TrainingSample> ConceptTrainer::generate_samples(
        ConceptId cid,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm) {

    std::vector<TrainingSample> samples;
    std::unordered_set<ConceptId> connected;

    static const size_t RECALL_HASH = std::hash<std::string>{}("recall");

    // Epistemic trust factor per EpistemicType (Convergence v2, Section 7.1)
    auto epistemic_trust = [](EpistemicType type) -> float {
        switch (type) {
            case EpistemicType::FACT:        return 1.0f;
            case EpistemicType::DEFINITION:  return 1.0f;
            case EpistemicType::THEORY:      return 0.7f;
            case EpistemicType::INFERENCE:   return 0.6f;
            case EpistemicType::HYPOTHESIS:  return 0.4f;
            case EpistemicType::SPECULATION: return 0.2f;
        }
        return 0.5f;
    };

    // Get source concept's epistemic type
    auto source_info = ltm.retrieve_concept(cid);
    float source_trust = source_info ? epistemic_trust(source_info->epistemic.type) : 0.5f;

    auto outgoing = ltm.get_outgoing_relations(cid);
    for (const auto& rel : outgoing) {
        connected.insert(rel.target);

        // Epistemic-weighted target (Convergence v2, Section 7.1)
        auto target_info = ltm.retrieve_concept(rel.target);
        if (!target_info) continue;

        // Anti-Knowledge: simple AK = skip, complex AK = negative training signal
        if (target_info->is_anti_knowledge && target_info->complexity_score < 0.3f) continue;

        float target_trust = epistemic_trust(target_info->epistemic.type);
        float epistemic_factor = source_trust * target_trust;

        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(rel.type);
        sample.context_embedding = embeddings.make_target_embedding(RECALL_HASH, cid, rel.target);
        double target_val = rel.weight * epistemic_factor;
        // Complex AK: negate target to teach "this path is wrong"
        if (target_info->is_anti_knowledge) target_val = -target_val;
        sample.target = target_val;
        sample.weight = static_cast<double>(epistemic_factor);
        samples.push_back(sample);
    }

    auto incoming = ltm.get_incoming_relations(cid);
    for (const auto& rel : incoming) {
        connected.insert(rel.source);

        auto incoming_info = ltm.retrieve_concept(rel.source);
        if (!incoming_info) continue;

        // Anti-Knowledge: simple AK = skip, complex AK = negative training signal
        if (incoming_info->is_anti_knowledge && incoming_info->complexity_score < 0.3f) continue;

        float incoming_trust = epistemic_trust(incoming_info->epistemic.type);
        float epistemic_factor = source_trust * incoming_trust;

        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(rel.type);
        sample.context_embedding = embeddings.make_target_embedding(RECALL_HASH, cid, rel.source);
        double target_val = rel.weight * config_.incoming_discount * epistemic_factor;
        if (incoming_info->is_anti_knowledge) target_val = -target_val;
        sample.target = target_val;
        sample.weight = static_cast<double>(epistemic_factor);
        samples.push_back(sample);
    }

    size_t num_positives = samples.size();
    if (num_positives == 0) return samples;

    auto all_ids = ltm.get_all_concept_ids();
    size_t num_negatives_needed = num_positives * config_.neg_ratio;

    size_t seed = static_cast<size_t>(cid) * 2654435761u;
    size_t neg_count = 0;

    for (size_t attempt = 0; attempt < all_ids.size() * 2 && neg_count < num_negatives_needed; ++attempt) {
        size_t mixed = (seed ^ (attempt * 6364136223846793005u)) + 1442695040888963407u;
        size_t idx = mixed % all_ids.size();
        ConceptId candidate = all_ids[idx];

        if (candidate == cid || connected.count(candidate) > 0) continue;

        RelationType neg_type = static_cast<RelationType>(neg_count % 10);
        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(neg_type);
        sample.context_embedding = embeddings.make_target_embedding(RECALL_HASH, cid, candidate);
        sample.target = config_.neg_target;
        samples.push_back(sample);
        ++neg_count;
    }

    return samples;
}

// =============================================================================
// Single model training
// =============================================================================

MicroTrainingResult ConceptTrainer::train_single(
        ConceptId cid,
        ConceptModel& model,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm) {

    auto samples = generate_samples(cid, embeddings, ltm);

    // Append multi-hop samples
    auto multihop = multihop_sampler_.generate_samples(cid, embeddings, ltm);
    samples.insert(samples.end(), multihop.begin(), multihop.end());

    return model.train(samples, config_.model_config);
}

// =============================================================================
// Batch training
// =============================================================================

ConceptTrainerStats ConceptTrainer::train_all(
        ConceptModelRegistry& registry,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm) {

    auto model_ids = registry.get_model_ids();
    const auto& concept_store = embeddings.concept_embeddings();
    static const size_t RECALL_HASH = std::hash<std::string>{}("recall");

    // =========================================================================
    // Phase 0: Discover patterns (if PatternDiscovery available)
    // =========================================================================

    std::vector<DiscoveredPattern> patterns;
    // Per-concept index: concept → list of pattern pointers
    std::unordered_map<ConceptId, std::vector<const DiscoveredPattern*>> pattern_index;
    if (pattern_discovery_) {
        patterns = pattern_discovery_->discover_all();
        for (const auto& pat : patterns) {
            for (ConceptId cid : pat.involved_concepts) {
                pattern_index[cid].push_back(&pat);
            }
        }
    }

    // =========================================================================
    // Phase 1 (sequential): Pre-compute all data from shared state
    // =========================================================================
    // All LTM/EmbeddingManager access happens here — no thread-safety issues.

    struct RefinedInput {
        FlexEmbedding rel_emb;
        FlexEmbedding ctx_emb;
        FlexEmbedding concept_from;
        FlexEmbedding concept_to;
        double target;
    };

    struct PrecomputedData {
        ConceptId cid = 0;
        std::vector<TrainingSample> samples;
        size_t num_positives = 0;
        std::vector<RefinedInput> refined_inputs;
        // Stats tracking
        size_t multihop_count = 0;
        size_t pattern_count = 0;
        double total_path_depth = 0.0;
        size_t path_count = 0;
    };

    std::vector<PrecomputedData> all_data;
    all_data.reserve(model_ids.size());

    for (ConceptId cid : model_ids) {
        ConceptModel* model = registry.get_model(cid);
        if (!model) continue;

        PrecomputedData data;
        data.cid = cid;
        data.samples = generate_samples(cid, embeddings, ltm);

        // Append multi-hop samples
        auto multihop_paths = multihop_sampler_.extract_paths(cid, ltm);
        auto multihop_samples = multihop_sampler_.generate_samples(cid, embeddings, ltm);
        data.multihop_count = multihop_samples.size();
        for (const auto& p : multihop_paths) {
            data.total_path_depth += static_cast<double>(p.edges.size());
            data.path_count++;
        }
        data.samples.insert(data.samples.end(), multihop_samples.begin(), multihop_samples.end());

        // Direct pattern training: patterns ARE the data
        if (auto pit = pattern_index.find(cid); pit != pattern_index.end()) {
            size_t gap_n = 0, cluster_n = 0, cycle_n = 0, bridge_n = 0;
            for (const auto* pat : pit->second) {
                // Find a target concept in this pattern
                ConceptId target = 0;
                bool found = false;
                for (ConceptId pid : pat->involved_concepts) {
                    if (pid != cid) { target = pid; found = true; break; }
                }
                if (!found) continue;

                TrainingSample ps;
                ps.context_embedding = embeddings.make_target_embedding(RECALL_HASH, cid, target);

                if (pat->pattern_type == "gap" && gap_n < 5) {
                    // Gap: concept[0] probably relates to concept[1] via the inferred type
                    ps.relation_embedding = embeddings.get_relation_embedding(pat->gap_rel_type);
                    ps.target = 0.3 * pat->confidence;
                    data.samples.push_back(std::move(ps));
                    ++gap_n; ++data.pattern_count;
                } else if (pat->pattern_type == "cluster" && cluster_n < 3) {
                    ps.relation_embedding = embeddings.get_relation_embedding(RelationType::ASSOCIATED_WITH);
                    ps.target = pat->confidence * 0.4;
                    data.samples.push_back(std::move(ps));
                    ++cluster_n; ++data.pattern_count;
                } else if (pat->pattern_type == "cycle" && cycle_n < 2) {
                    ps.relation_embedding = embeddings.get_relation_embedding(RelationType::ASSOCIATED_WITH);
                    ps.target = 0.7 * pat->confidence;
                    data.samples.push_back(std::move(ps));
                    ++cycle_n; ++data.pattern_count;
                } else if (pat->pattern_type == "bridge" && bridge_n < 2) {
                    ps.relation_embedding = embeddings.get_relation_embedding(RelationType::ASSOCIATED_WITH);
                    ps.target = 0.5 * pat->confidence;
                    data.samples.push_back(std::move(ps));
                    ++bridge_n; ++data.pattern_count;
                }
            }
        }

        for (const auto& s : data.samples) {
            if (s.target > config_.neg_target) data.num_positives++;
        }

        if (data.num_positives > 0) {
            FlexEmbedding concept_from = concept_store.get_or_default(cid);
            auto outgoing = ltm.get_outgoing_relations(cid);

            // Reserve for direct + multi-hop refined inputs
            data.refined_inputs.reserve(outgoing.size() + multihop_paths.size());

            for (const auto& rel : outgoing) {
                RefinedInput ri;
                ri.concept_from = concept_from;
                ri.concept_to = concept_store.get_or_default(rel.target);
                ri.rel_emb = embeddings.get_relation_embedding(rel.type);
                ri.ctx_emb = embeddings.make_target_embedding(RECALL_HASH, cid, rel.target);
                ri.target = rel.weight;
                data.refined_inputs.push_back(std::move(ri));
            }

            // Multi-hop paths as refined inputs
            for (const auto& path : multihop_paths) {
                RefinedInput ri;
                ri.concept_from = concept_from;
                ri.concept_to = concept_store.get_or_default(path.terminus);
                ri.rel_emb = multihop_sampler_.compose_path_embedding(path.edges, embeddings);
                ri.ctx_emb = embeddings.make_target_embedding(RECALL_HASH, cid, path.terminus);
                ri.target = path.path_weight;
                data.refined_inputs.push_back(std::move(ri));
            }
        }

        all_data.push_back(std::move(data));
    }

    // =========================================================================
    // Phase 2 (parallel): Train models using only pre-computed data
    // =========================================================================
    // Each thread only touches its own ConceptModel — no shared mutable state.

    size_t num_threads = std::min(
        static_cast<size_t>(std::thread::hardware_concurrency()), size_t{8});
    if (num_threads == 0) num_threads = 1;

    struct ThreadStats {
        size_t models_trained = 0;
        size_t total_samples = 0;
        size_t total_epochs = 0;
        double total_loss = 0.0;
        size_t models_converged = 0;
        size_t refined_updates = 0;
        size_t multihop_samples = 0;
        size_t pattern_samples = 0;
        double total_path_depth = 0.0;
        size_t path_count = 0;
    };
    std::vector<ThreadStats> thread_stats(num_threads);

    auto worker = [&](size_t tid, size_t start, size_t end) {
        auto& ts = thread_stats[tid];
        RefinedAdamState adam_state;  // One per thread, reused across models

        for (size_t i = start; i < end; ++i) {
            auto& data = all_data[i];
            ConceptModel* model = registry.get_model(data.cid);
            if (!model) continue;

            ts.total_samples += data.samples.size();
            ts.multihop_samples += data.multihop_count;
            ts.pattern_samples += data.pattern_count;
            ts.total_path_depth += data.total_path_depth;
            ts.path_count += data.path_count;

            if (data.num_positives == 0) continue;

            auto result = model->train(data.samples, config_.model_config);
            ts.models_trained++;
            ts.total_epochs += result.epochs_run;
            ts.total_loss += result.final_loss;
            if (result.converged) ts.models_converged++;

            // Refined training with Adam (multiple epochs)
            adam_state.reset();
            for (size_t epoch = 0; epoch < config_.refined_epochs; ++epoch) {
                for (const auto& ri : data.refined_inputs) {
                    model->train_refined(ri.rel_emb, ri.ctx_emb, ri.concept_from,
                                         ri.concept_to, ri.target,
                                         config_.kan_learning_rate, adam_state);
                }
            }
            ts.refined_updates += data.refined_inputs.size() * config_.refined_epochs;
        }
    };

    std::vector<std::thread> threads;
    size_t chunk = (all_data.size() + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, all_data.size());
        if (start >= end) break;
        threads.emplace_back(worker, t, start, end);
    }
    for (auto& t : threads) t.join();

    // Merge per-thread stats
    ConceptTrainerStats stats;
    double total_loss = 0.0;
    double total_path_depth = 0.0;
    size_t total_path_count = 0;
    for (const auto& ts : thread_stats) {
        stats.models_trained += ts.models_trained;
        stats.total_samples += ts.total_samples;
        stats.total_epochs += ts.total_epochs;
        total_loss += ts.total_loss;
        stats.models_converged += ts.models_converged;
        stats.refined_updates += ts.refined_updates;
        stats.multihop_samples += ts.multihop_samples;
        stats.pattern_samples += ts.pattern_samples;
        total_path_depth += ts.total_path_depth;
        total_path_count += ts.path_count;
    }
    if (stats.models_trained > 0) {
        stats.avg_final_loss = total_loss / static_cast<double>(stats.models_trained);
    }
    if (total_path_count > 0) {
        stats.avg_path_depth = total_path_depth / static_cast<double>(total_path_count);
    }

    return stats;
}

// =============================================================================
// KAN training from validation
// =============================================================================

void ConceptTrainer::train_kan_from_validation(
        ConceptModel& model,
        const std::vector<std::pair<double, double>>& bilinear_vs_validated) {

    // Legacy path: train FlexKAN with bilinear_score only (no concept embeddings available)
    for (size_t epoch = 0; epoch < config_.kan_epochs; ++epoch) {
        for (const auto& [bilinear_score, validated_target] : bilinear_vs_validated) {
            model.train_kan(bilinear_score, 0.5, validated_target, config_.kan_learning_rate);
        }
    }
}

} // namespace brain19
