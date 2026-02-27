#pragma once

#include "concept_model.hpp"
#include "concept_model_registry.hpp"
#include "multihop_sampler.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../evolution/pattern_discovery.hpp"

#include <unordered_map>
#include <vector>
#include <cstddef>

namespace brain19 {

// =============================================================================
// CONCEPT TRAINER
// =============================================================================
//
// Trains ConceptModels from KG structure. Extends MicroTrainer with KAN training.
// Supports multi-hop path samples and pattern-driven samples.
//

struct ConceptTrainerConfig {
    MicroTrainingConfig model_config;
    double incoming_discount = 0.8;
    size_t neg_ratio = 3;
    double neg_target = 0.05;
    // Refined training (MultiHead + KAN)
    double kan_learning_rate = 0.005;
    size_t kan_epochs = 50;
    size_t refined_epochs = 10;  // Epochs over refined data per model

    // Multi-hop training
    MultiHopConfig multihop_config;

    ConceptTrainerConfig() {
        model_config.max_epochs = 500;
        model_config.convergence_threshold = 1e-4;
    }
};

struct ConceptTrainerStats {
    size_t models_trained = 0;
    size_t total_samples = 0;
    size_t total_epochs = 0;
    double avg_final_loss = 0.0;
    size_t models_converged = 0;
    size_t kan_updates = 0;
    size_t refined_updates = 0;
    size_t models_rolled_back = 0;  // Validation gate rollbacks
    // Multi-hop & pattern stats
    size_t multihop_samples = 0;
    size_t pattern_samples = 0;
    double avg_path_depth = 0.0;
};

class ConceptTrainer {
public:
    explicit ConceptTrainer(const ConceptTrainerConfig& config = ConceptTrainerConfig{});

    // Set optional PatternDiscovery source for pattern-driven training
    void set_pattern_discovery(PatternDiscovery* pd) { pattern_discovery_ = pd; }

    ConceptTrainerStats train_all(ConceptModelRegistry& registry,
                                  EmbeddingManager& embeddings,
                                  const LongTermMemory& ltm);

    MicroTrainingResult train_single(ConceptId cid, ConceptModel& model,
                                      EmbeddingManager& embeddings,
                                      const LongTermMemory& ltm);

    std::vector<TrainingSample> generate_samples(ConceptId cid,
                                                  EmbeddingManager& embeddings,
                                                  const LongTermMemory& ltm);

    // Train KAN from validation results: pairs of (bilinear_score, validated_target)
    void train_kan_from_validation(ConceptModel& model,
                                    const std::vector<std::pair<double, double>>& bilinear_vs_validated);

private:
    ConceptTrainerConfig config_;
    MultiHopSampler multihop_sampler_;
    PatternDiscovery* pattern_discovery_ = nullptr;
};

} // namespace brain19
