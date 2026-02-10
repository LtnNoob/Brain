#include "micro_trainer.hpp"

#include <algorithm>
#include <unordered_set>

namespace brain19 {

MicroTrainer::MicroTrainer(const TrainerConfig& config)
    : config_(config)
{}

// =============================================================================
// Sample generation
// =============================================================================

std::vector<TrainingSample> MicroTrainer::generate_samples(
        ConceptId cid,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm) {

    std::vector<TrainingSample> samples;
    std::unordered_set<ConceptId> connected;

    const Vec10& recall_ctx = embeddings.recall_context();

    // Positive samples from outgoing relations
    auto outgoing = ltm.get_outgoing_relations(cid);
    for (const auto& rel : outgoing) {
        connected.insert(rel.target);
        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(rel.type);
        sample.context_embedding = recall_ctx;
        sample.target = rel.weight;
        samples.push_back(sample);
    }

    // Positive samples from incoming relations (discounted)
    auto incoming = ltm.get_incoming_relations(cid);
    for (const auto& rel : incoming) {
        connected.insert(rel.source);
        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(rel.type);
        sample.context_embedding = recall_ctx;
        sample.target = rel.weight * config_.incoming_discount;
        samples.push_back(sample);
    }

    size_t num_positives = samples.size();
    if (num_positives == 0) {
        return samples;  // No relations, no training data
    }

    // Negative samples from non-connected concepts
    auto all_ids = ltm.get_all_concept_ids();
    size_t num_negatives_needed = num_positives * config_.neg_ratio;

    // Deterministic pseudo-random selection using concept ID as seed
    size_t seed = static_cast<size_t>(cid) * 2654435761u;
    size_t neg_count = 0;

    for (size_t attempt = 0; attempt < all_ids.size() * 2 && neg_count < num_negatives_needed; ++attempt) {
        // Pseudo-random index selection
        size_t mixed = (seed ^ (attempt * 6364136223846793005u)) + 1442695040888963407u;
        size_t idx = mixed % all_ids.size();
        ConceptId candidate = all_ids[idx];

        // Skip self and connected concepts
        if (candidate == cid || connected.count(candidate) > 0) {
            continue;
        }

        // Use a varied relation type for negatives (cycle through types)
        RelationType neg_type = static_cast<RelationType>(neg_count % NUM_RELATION_TYPES);
        TrainingSample sample;
        sample.relation_embedding = embeddings.get_relation_embedding(neg_type);
        sample.context_embedding = recall_ctx;
        sample.target = config_.neg_target;
        samples.push_back(sample);
        ++neg_count;
    }

    return samples;
}

// =============================================================================
// Single model training
// =============================================================================

TrainingResult MicroTrainer::train_single(
        ConceptId cid,
        MicroModel& model,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm) {

    auto samples = generate_samples(cid, embeddings, ltm);
    return model.train(samples, config_.model_config);
}

// =============================================================================
// Batch training
// =============================================================================

TrainerStats MicroTrainer::train_all(
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm) {

    TrainerStats stats;
    auto model_ids = registry.get_model_ids();
    double total_loss = 0.0;

    for (ConceptId cid : model_ids) {
        MicroModel* model = registry.get_model(cid);
        if (!model) continue;

        auto samples = generate_samples(cid, embeddings, ltm);
        stats.total_samples += samples.size();

        if (samples.empty()) continue;

        auto result = model->train(samples, config_.model_config);
        stats.models_trained++;
        stats.total_epochs += result.epochs_run;
        total_loss += result.final_loss;
        if (result.converged) {
            stats.models_converged++;
        }
    }

    if (stats.models_trained > 0) {
        stats.avg_final_loss = total_loss / static_cast<double>(stats.models_trained);
    }

    return stats;
}

} // namespace brain19
