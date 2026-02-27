#include "chain_kan.hpp"
#include "../micromodel/micro_model.hpp"  // sigmoid

#include <algorithm>
#include <cmath>

namespace brain19 {

// =============================================================================
// Tent basis (same as FlexKAN in concept_model.cpp)
// =============================================================================

static double ck_tent_basis(double x, size_t k, size_t num_knots) {
    double center = static_cast<double>(k) / static_cast<double>(num_knots - 1);
    double width = 1.0 / static_cast<double>(num_knots - 1);
    double dist = std::abs(x - center);
    if (dist >= width) return 0.0;
    return 1.0 - dist / width;
}

static double ck_tent_derivative(double x, size_t k, size_t num_knots) {
    double center = static_cast<double>(k) / static_cast<double>(num_knots - 1);
    double width = 1.0 / static_cast<double>(num_knots - 1);
    double diff = x - center;
    if (std::abs(diff) >= width) return 0.0;
    if (diff > 0.0) return -1.0 / width;
    if (diff < 0.0) return  1.0 / width;
    return 0.0;
}

static double ck_edge_forward(const double* coeffs, double x, size_t num_knots) {
    x = std::max(0.0, std::min(1.0, x));
    double result = 0.0;
    for (size_t k = 0; k < num_knots; ++k) {
        result += coeffs[k] * ck_tent_basis(x, k, num_knots);
    }
    return result;
}

// =============================================================================
// initialize — Xavier for projection, identity-passthrough for KAN
// =============================================================================

void ChainKAN::initialize() {
    // Xavier init for projection: scale = sqrt(2 / (fan_in + fan_out))
    double scale = std::sqrt(2.0 / static_cast<double>(PROJ_IN + PROJ_OUT));
    for (size_t i = 0; i < PROJ_W_SIZE; ++i) {
        double x = std::sin(static_cast<double>(i * 37 + 53)) * 43758.5453;
        x = x - std::floor(x);
        proj_W[i] = (x * 2.0 - 1.0) * scale;
    }
    proj_b.fill(0.0);

    // KAN: identity initialization (output ≈ 0.5 initially)
    // Same pattern as FlexKAN::initialize_identity but for input[0]→hidden[0]→output
    kan_params.fill(0.0);

    auto safe_logit = [](double p) -> double {
        p = std::max(0.01, std::min(0.99, p));
        return std::log(p / (1.0 - p));
    };

    // Layer 0: edge from input_0 to hidden_0
    size_t l0_edge = (0 * KAN_HIDDEN + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        kan_params[l0_edge + k] = safe_logit(x);
    }

    // Layer 1: edge from hidden_0 to output
    size_t l1_edge = (LAYER0_EDGES + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        kan_params[l1_edge + k] = safe_logit(x);
    }
}

// =============================================================================
// evaluate — project 64→6, then KAN forward
// =============================================================================

double ChainKAN::evaluate(const double* prev_state_32, const double* new_state_32) const {
    // Concatenate inputs: [prev(32), new(32)] = 64D
    double concat[PROJ_IN];
    for (size_t i = 0; i < 32; ++i) {
        concat[i] = prev_state_32[i];
        concat[32 + i] = new_state_32[i];
    }

    // Linear projection 64→6 with sigmoid activation
    double proj[PROJ_OUT];
    for (size_t i = 0; i < PROJ_OUT; ++i) {
        double sum = proj_b[i];
        for (size_t j = 0; j < PROJ_IN; ++j) {
            sum += proj_W[i * PROJ_IN + j] * concat[j];
        }
        proj[i] = sigmoid(sum);  // clamps to (0,1) for KAN input
    }

    // KAN Layer 0: 6→4
    double hidden[KAN_HIDDEN] = {};
    for (size_t j = 0; j < KAN_HIDDEN; ++j) {
        for (size_t i = 0; i < KAN_INPUT; ++i) {
            size_t edge_idx = (i * KAN_HIDDEN + j) * NUM_KNOTS;
            hidden[j] += ck_edge_forward(&kan_params[edge_idx], proj[i], NUM_KNOTS);
        }
    }

    // Activate hidden
    for (size_t j = 0; j < KAN_HIDDEN; ++j) {
        hidden[j] = sigmoid(hidden[j]);
    }

    // KAN Layer 1: 4→1
    double output_raw = 0.0;
    for (size_t j = 0; j < KAN_HIDDEN; ++j) {
        size_t edge_idx = (LAYER0_EDGES + j) * NUM_KNOTS;
        output_raw += ck_edge_forward(&kan_params[edge_idx], hidden[j], NUM_KNOTS);
    }

    return sigmoid(output_raw);
}

// =============================================================================
// train — MSE loss, analytical gradient through KAN + projection, SGD
// =============================================================================

void ChainKAN::train(const std::vector<Sample>& samples, double lr, size_t epochs) {
    if (samples.empty()) return;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& s : samples) {
            // ── Forward pass with cached intermediates ──

            double concat[PROJ_IN];
            for (size_t i = 0; i < 32; ++i) {
                concat[i] = s.prev[i];
                concat[32 + i] = s.next[i];
            }

            // Projection forward
            double proj_act[PROJ_OUT];
            for (size_t i = 0; i < PROJ_OUT; ++i) {
                double sum = proj_b[i];
                for (size_t j = 0; j < PROJ_IN; ++j) {
                    sum += proj_W[i * PROJ_IN + j] * concat[j];
                }
                proj_act[i] = sigmoid(sum);
            }

            // KAN input (clamped projection output)
            double kan_in[KAN_INPUT];
            for (size_t i = 0; i < KAN_INPUT; ++i) {
                kan_in[i] = std::max(0.0, std::min(1.0, proj_act[i]));
            }

            // KAN Layer 0
            double hidden_raw[KAN_HIDDEN] = {};
            for (size_t j = 0; j < KAN_HIDDEN; ++j) {
                for (size_t i = 0; i < KAN_INPUT; ++i) {
                    size_t edge_idx = (i * KAN_HIDDEN + j) * NUM_KNOTS;
                    hidden_raw[j] += ck_edge_forward(&kan_params[edge_idx], kan_in[i], NUM_KNOTS);
                }
            }

            double hidden_act[KAN_HIDDEN];
            for (size_t j = 0; j < KAN_HIDDEN; ++j) {
                hidden_act[j] = sigmoid(hidden_raw[j]);
            }

            // KAN Layer 1
            double output_raw = 0.0;
            for (size_t j = 0; j < KAN_HIDDEN; ++j) {
                size_t edge_idx = (LAYER0_EDGES + j) * NUM_KNOTS;
                output_raw += ck_edge_forward(&kan_params[edge_idx], hidden_act[j], NUM_KNOTS);
            }

            double output = sigmoid(output_raw);

            // ── Backward pass ──

            // MSE gradient: d/d(output) of 0.5*(output-target)^2
            double dL_dout = output - s.target;
            double dL_dout_raw = dL_dout * output * (1.0 - output);

            // KAN Layer 1 backward
            double dL_dhidden_act[KAN_HIDDEN] = {};
            for (size_t j = 0; j < KAN_HIDDEN; ++j) {
                size_t edge_base = (LAYER0_EDGES + j) * NUM_KNOTS;

                // Update layer 1 coefficients
                for (size_t k = 0; k < NUM_KNOTS; ++k) {
                    double grad = dL_dout_raw * ck_tent_basis(hidden_act[j], k, NUM_KNOTS);
                    kan_params[edge_base + k] -= lr * grad;
                }

                // Backprop to hidden_act
                double d_edge_dx = 0.0;
                for (size_t k = 0; k < NUM_KNOTS; ++k) {
                    d_edge_dx += kan_params[edge_base + k] * ck_tent_derivative(hidden_act[j], k, NUM_KNOTS);
                }
                dL_dhidden_act[j] = dL_dout_raw * d_edge_dx;
            }

            // Through sigmoid
            double dL_dhidden_raw[KAN_HIDDEN];
            for (size_t j = 0; j < KAN_HIDDEN; ++j) {
                dL_dhidden_raw[j] = dL_dhidden_act[j] * hidden_act[j] * (1.0 - hidden_act[j]);
            }

            // KAN Layer 0 backward + backprop to projection output
            double dL_dkan_in[KAN_INPUT] = {};
            for (size_t i = 0; i < KAN_INPUT; ++i) {
                for (size_t j = 0; j < KAN_HIDDEN; ++j) {
                    size_t edge_base = (i * KAN_HIDDEN + j) * NUM_KNOTS;

                    // Update layer 0 coefficients
                    for (size_t k = 0; k < NUM_KNOTS; ++k) {
                        double grad = dL_dhidden_raw[j] * ck_tent_basis(kan_in[i], k, NUM_KNOTS);
                        kan_params[edge_base + k] -= lr * grad;
                    }

                    // Backprop to kan_in
                    double d_edge_dx = 0.0;
                    for (size_t k = 0; k < NUM_KNOTS; ++k) {
                        d_edge_dx += kan_params[edge_base + k] * ck_tent_derivative(kan_in[i], k, NUM_KNOTS);
                    }
                    dL_dkan_in[i] += dL_dhidden_raw[j] * d_edge_dx;
                }
            }

            // Through projection sigmoid: dL/d(proj_raw[i])
            double dL_dproj_raw[PROJ_OUT];
            for (size_t i = 0; i < PROJ_OUT; ++i) {
                dL_dproj_raw[i] = dL_dkan_in[i] * proj_act[i] * (1.0 - proj_act[i]);
            }

            // Update projection weights and biases
            for (size_t i = 0; i < PROJ_OUT; ++i) {
                proj_b[i] -= lr * dL_dproj_raw[i];
                for (size_t j = 0; j < PROJ_IN; ++j) {
                    proj_W[i * PROJ_IN + j] -= lr * dL_dproj_raw[i] * concat[j];
                }
            }
        }
    }
}

} // namespace brain19
