#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace brain19 {

// =============================================================================
// ChainKAN — Shared KAN for chain coherence evaluation
// =============================================================================
//
// Evaluates whether a chain state transition is coherent.
// Input: prev_state(32) ⊕ new_state(32) = 64D
// Pipeline: Linear projection 64→6, then FlexKAN [6,4,1] with 10 knots
//
// Total parameters: 390 (projection) + 280 (KAN) = 670
//

struct ChainKAN {
    // Projection: 64→6
    static constexpr size_t PROJ_IN = 64;
    static constexpr size_t PROJ_OUT = 6;
    static constexpr size_t PROJ_W_SIZE = PROJ_IN * PROJ_OUT;  // 384
    std::array<double, PROJ_W_SIZE> proj_W{};
    std::array<double, PROJ_OUT> proj_b{};

    // FlexKAN [6,4,1] with 10 knots (same architecture as ConceptModel's FlexKAN)
    static constexpr size_t KAN_INPUT = 6;
    static constexpr size_t KAN_HIDDEN = 4;
    static constexpr size_t NUM_KNOTS = 10;
    static constexpr size_t LAYER0_EDGES = KAN_INPUT * KAN_HIDDEN;       // 24
    static constexpr size_t LAYER1_EDGES = KAN_HIDDEN * 1;               // 4
    static constexpr size_t TOTAL_KAN_EDGES = LAYER0_EDGES + LAYER1_EDGES; // 28
    static constexpr size_t KAN_PARAMS = TOTAL_KAN_EDGES * NUM_KNOTS;    // 280

    std::array<double, KAN_PARAMS> kan_params{};

    // Total: 384 + 6 + 280 = 670 params

    void initialize();

    // Evaluate chain coherence: returns value in (0, 1)
    double evaluate(const double* prev_state_32, const double* new_state_32) const;

    // Self-supervised training sample
    struct Sample {
        std::array<double, 32> prev{};
        std::array<double, 32> next{};
        double target = 0.0;
    };

    // Train on collected samples (MSE loss, SGD)
    void train(const std::vector<Sample>& samples, double lr, size_t epochs);
};

} // namespace brain19
