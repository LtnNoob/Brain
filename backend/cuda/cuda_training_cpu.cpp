// cuda/cuda_training_cpu.cpp — CPU stub (no-op, returns false)
#include "cuda_training.h"

namespace brain19 {
namespace cuda {

bool train_sgd_gpu(const TrainingData&, TrainingWeights&,
                   const TrainingConfig&, TrainingResult&) {
    return false;  // No GPU available
}

bool train_sgd_v11_gpu(const TrainingData&, TrainingWeights&,
                       const TrainingConfig&, TrainingResult&) {
    return false;  // No GPU available
}

bool train_deep_kan_gpu(const TrainingData&, DeepKANWeights&,
                        const DeepKANConfig&, TrainingResult&) {
    return false;  // No GPU available
}

} // namespace cuda
} // namespace brain19
