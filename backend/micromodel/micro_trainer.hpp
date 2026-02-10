#pragma once

#include "micro_model.hpp"
#include "micro_model_registry.hpp"
#include "embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"

#include <vector>
#include <cstddef>

namespace brain19 {

// =============================================================================
// MICRO-TRAINER
// =============================================================================
//
// Bootstraps training data from KG structure and trains MicroModels.
//
// Training data generation for concept C:
//   Positives: outgoing relations (C→T, type, weight) → target = weight
//              incoming relations (T→C) → target = weight * 0.8
//   Negatives: 3× negatives per positive, from non-connected concepts, target ≈ 0.0
//

struct TrainerConfig {
    TrainingConfig model_config;    // Per-model training config
    double incoming_discount = 0.8; // Weight discount for incoming relations
    size_t neg_ratio = 3;           // Negatives per positive sample
    double neg_target = 0.05;       // Target for negative samples

    TrainerConfig() = default;
};

struct TrainerStats {
    size_t models_trained = 0;
    size_t total_samples = 0;
    size_t total_epochs = 0;
    double avg_final_loss = 0.0;
    size_t models_converged = 0;
};

class MicroTrainer {
public:
    explicit MicroTrainer(const TrainerConfig& config = TrainerConfig{});

    // Train all models in the registry from KG data
    TrainerStats train_all(MicroModelRegistry& registry,
                           EmbeddingManager& embeddings,
                           const LongTermMemory& ltm);

    // Train a single model
    TrainingResult train_single(ConceptId cid,
                                MicroModel& model,
                                EmbeddingManager& embeddings,
                                const LongTermMemory& ltm);

    // Generate training samples for a concept
    std::vector<TrainingSample> generate_samples(ConceptId cid,
                                                  EmbeddingManager& embeddings,
                                                  const LongTermMemory& ltm);

private:
    TrainerConfig config_;
};

} // namespace brain19
