#include "concept_trainer.hpp"

#include <algorithm>
#include <thread>
#include <unordered_set>

namespace brain19 {

ConceptTrainer::ConceptTrainer(const ConceptTrainerConfig& config)
    : config_(config)
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

    auto outgoing = ltm.get_outgoing_relations(cid);
    for (const auto& rel : outgoing) {
        connected.insert(rel.target);
        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(rel.type);
        sample.context_embedding = embeddings.make_target_embedding(RECALL_HASH, cid, rel.target);
        sample.target = rel.weight;
        samples.push_back(sample);
    }

    auto incoming = ltm.get_incoming_relations(cid);
    for (const auto& rel : incoming) {
        connected.insert(rel.source);
        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(rel.type);
        sample.context_embedding = embeddings.make_target_embedding(RECALL_HASH, cid, rel.source);
        sample.target = rel.weight * config_.incoming_discount;
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
    };

    std::vector<PrecomputedData> all_data;
    all_data.reserve(model_ids.size());

    for (ConceptId cid : model_ids) {
        ConceptModel* model = registry.get_model(cid);
        if (!model) continue;

        PrecomputedData data;
        data.cid = cid;
        data.samples = generate_samples(cid, embeddings, ltm);

        for (const auto& s : data.samples) {
            if (s.target > config_.neg_target) data.num_positives++;
        }

        if (data.num_positives > 0) {
            FlexEmbedding concept_from = concept_store.get_or_default(cid);
            auto outgoing = ltm.get_outgoing_relations(cid);
            data.refined_inputs.reserve(outgoing.size());

            for (const auto& rel : outgoing) {
                RefinedInput ri;
                ri.concept_from = concept_from;
                ri.concept_to = concept_store.get_or_default(rel.target);
                ri.rel_emb = embeddings.get_relation_embedding(rel.type);
                ri.ctx_emb = embeddings.make_target_embedding(RECALL_HASH, cid, rel.target);
                ri.target = rel.weight;
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
            if (data.num_positives == 0) continue;

            auto result = model->train(data.samples, config_.model_config);
            ts.models_trained++;
            ts.total_epochs += result.epochs_run;
            ts.total_loss += result.final_loss;
            if (result.converged) ts.models_converged++;

            // Refined training with Adam
            adam_state.reset();
            for (const auto& ri : data.refined_inputs) {
                model->train_refined(ri.rel_emb, ri.ctx_emb, ri.concept_from,
                                     ri.concept_to, ri.target,
                                     config_.kan_learning_rate, adam_state);
                ts.refined_updates++;
            }
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
    for (const auto& ts : thread_stats) {
        stats.models_trained += ts.models_trained;
        stats.total_samples += ts.total_samples;
        stats.total_epochs += ts.total_epochs;
        total_loss += ts.total_loss;
        stats.models_converged += ts.models_converged;
        stats.refined_updates += ts.refined_updates;
    }
    if (stats.models_trained > 0) {
        stats.avg_final_loss = total_loss / static_cast<double>(stats.models_trained);
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
