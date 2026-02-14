#include "concept_model.hpp"

#include <algorithm>
#include <numeric>

namespace brain19 {

// =============================================================================
// Static helpers
// =============================================================================

static std::array<double, 4> cyclic_compress(const std::vector<double>& detail) {
    std::array<double, 4> compressed{};
    for (size_t d = 0; d < detail.size(); ++d) {
        compressed[d % 4] += detail[d];
    }
    return compressed;
}

static std::array<double, 20> make_input_vec(const FlexEmbedding& emb) {
    std::array<double, 20> input{};
    for (size_t i = 0; i < CORE_DIM; ++i) {
        input[i] = emb.core[i];
    }
    auto compressed = cyclic_compress(emb.detail);
    for (size_t i = 0; i < 4; ++i) {
        input[CORE_DIM + i] = compressed[i];
    }
    return input;
}

// =============================================================================
// MultiHeadBilinear
// =============================================================================

void MultiHeadBilinear::compute(const FlexEmbedding& e_q, const FlexEmbedding& e_k,
                                 std::array<double, K>& scores) const {
    auto input_q = make_input_vec(e_q);
    auto input_k = make_input_vec(e_k);

    for (size_t h = 0; h < K; ++h) {
        // P_i at offset h * PARAMS_PER_HEAD, size D_PROJ x INPUT_DIM
        // Q_i at offset h * PARAMS_PER_HEAD + D_PROJ * INPUT_DIM
        size_t p_offset = h * PARAMS_PER_HEAD;
        size_t q_offset = p_offset + D_PROJ * INPUT_DIM;

        double dot = 0.0;
        for (size_t d = 0; d < D_PROJ; ++d) {
            double pq = 0.0;
            double pk = 0.0;
            for (size_t j = 0; j < INPUT_DIM; ++j) {
                pq += params[p_offset + d * INPUT_DIM + j] * input_q[j];
                pk += params[q_offset + d * INPUT_DIM + j] * input_k[j];
            }
            dot += pq * pk;
        }
        scores[h] = dot;
    }
}

void MultiHeadBilinear::initialize() {
    params.fill(0.0);
}

// =============================================================================
// FlexKAN — Lightweight [6,4,1] B-spline network
// =============================================================================
//
// Layer 0: 6 inputs -> 4 hidden  (24 edges, each 10 params = 240)
// Layer 1: 4 hidden -> 1 output  (4 edges, each 10 params = 40)
// Total: 280 params
//
// Each edge function uses a simplified B-spline:
//   f(x) = sum_{k=0}^{9} coeff[k] * basis_k(x)
// where basis_k is a tent function centered at knot k/(NUM_KNOTS-1).
//

static double tent_basis(double x, size_t k, size_t num_knots) {
    double center = static_cast<double>(k) / static_cast<double>(num_knots - 1);
    double width = 1.0 / static_cast<double>(num_knots - 1);
    double dist = std::abs(x - center);
    if (dist >= width) return 0.0;
    return 1.0 - dist / width;
}

static double kan_edge_forward(const double* coeffs, double x, size_t num_knots) {
    x = std::max(0.0, std::min(1.0, x));
    double result = 0.0;
    for (size_t k = 0; k < num_knots; ++k) {
        result += coeffs[k] * tent_basis(x, k, num_knots);
    }
    return result;
}

double FlexKAN::evaluate(const std::array<double, INPUT_DIM>& input) const {
    // Clamp inputs to [0,1]
    double in[INPUT_DIM];
    for (size_t i = 0; i < INPUT_DIM; ++i) {
        in[i] = std::max(0.0, std::min(1.0, input[i]));
    }

    // Layer 0: INPUT_DIM inputs -> HIDDEN_DIM hidden
    // Edge layout: edge(input_i, hidden_j) at index (i * HIDDEN_DIM + j) * NUM_KNOTS
    double hidden[HIDDEN_DIM] = {};
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        for (size_t i = 0; i < INPUT_DIM; ++i) {
            size_t edge_idx = (i * HIDDEN_DIM + j) * NUM_KNOTS;
            hidden[j] += kan_edge_forward(&params[edge_idx], in[i], NUM_KNOTS);
        }
    }

    // Activate hidden layer with sigmoid
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        hidden[j] = sigmoid(hidden[j]);
    }

    // Layer 1: HIDDEN_DIM hidden -> 1 output
    double output = 0.0;
    for (size_t j = 0; j < HIDDEN_DIM; ++j) {
        size_t edge_idx = (LAYER0_EDGES + j) * NUM_KNOTS;
        output += kan_edge_forward(&params[edge_idx], hidden[j], NUM_KNOTS);
    }

    return sigmoid(output);
}

void FlexKAN::train_step(const std::array<double, INPUT_DIM>& input,
                          double target, double learning_rate) {
    constexpr double eps = 1e-5;
    double base_output = evaluate(input);
    double base_loss = 0.5 * (base_output - target) * (base_output - target);

    for (size_t i = 0; i < TOTAL_PARAMS; ++i) {
        double orig = params[i];
        params[i] = orig + eps;
        double plus_output = evaluate(input);
        double plus_loss = 0.5 * (plus_output - target) * (plus_output - target);
        double grad = (plus_loss - base_loss) / eps;
        params[i] = orig - learning_rate * grad;
    }
}

void FlexKAN::initialize_identity() {
    params.fill(0.0);

    // We want: output ~ bilinear_score (which is input[4])
    // Two sigmoids in the path, so use logit-scaled coefficients.
    auto safe_logit = [](double p) -> double {
        p = std::max(0.01, std::min(0.99, p));
        return std::log(p / (1.0 - p));
    };

    // Layer 0: edge from input_4 to hidden_0 with logit identity
    // Edge index: (4 * HIDDEN_DIM + 0) * NUM_KNOTS
    size_t l0_edge = (4 * HIDDEN_DIM + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        params[l0_edge + k] = safe_logit(x);
    }

    // Layer 1: edge from hidden_0 to output with logit identity
    size_t l1_edge = (LAYER0_EDGES + 0) * NUM_KNOTS;
    for (size_t k = 0; k < NUM_KNOTS; ++k) {
        double x = static_cast<double>(k) / static_cast<double>(NUM_KNOTS - 1);
        params[l1_edge + k] = safe_logit(x);
    }
    // All other edges are zero — only input_4->hidden_0->output contributes.
}

// =============================================================================
// ConceptModel Construction
// =============================================================================

ConceptModel::ConceptModel() {
    // Initialize W with small values (Xavier-like for 16x16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        for (size_t j = 0; j < CORE_DIM; ++j) {
            double diag = (i == j) ? 0.1 : 0.0;
            double off = 0.01 * std::sin(static_cast<double>(i * 10 + j));
            W_[i * CORE_DIM + j] = diag + off;
        }
    }

    for (size_t i = 0; i < CORE_DIM; ++i) {
        b_[i] = 0.01 * std::cos(static_cast<double>(i));
    }

    for (size_t i = 0; i < CORE_DIM; ++i) {
        e_init_[i] = 0.1 * std::sin(static_cast<double>(i * 3 + 1));
        c_init_[i] = 0.1 * std::cos(static_cast<double>(i * 7 + 2));
    }

    multihead_.initialize();
    kan_.initialize_identity();
    reserved_.fill(0.0);
}

// =============================================================================
// Forward pass — operates on Core dimensions only
// =============================================================================

double ConceptModel::predict(const FlexEmbedding& e, const FlexEmbedding& c) const {
    CoreVec v;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = b_[i];
        for (size_t j = 0; j < CORE_DIM; ++j) {
            sum += W_[i * CORE_DIM + j] * c.core[j];
        }
        v[i] = sum;
    }

    double z = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        z += e.core[i] * v[i];
    }

    return sigmoid(z);
}

double ConceptModel::predict_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                                      const FlexEmbedding& concept_from,
                                      const FlexEmbedding& concept_to) const {
    double bilinear = predict(rel_emb, ctx_emb);

    // Compute multi-head scores from concept embeddings
    std::array<double, MultiHeadBilinear::K> mh_scores;
    multihead_.compute(concept_from, concept_to, mh_scores);

    // dim_fraction: maturity signal from concept detail dimensions
    double dim_fraction = static_cast<double>(
        std::min(concept_from.detail.size(), concept_to.detail.size())) / 496.0;

    // Assemble KAN input: [s_0, s_1, s_2, s_3, bilinear_score, dim_fraction]
    std::array<double, FlexKAN::INPUT_DIM> kan_input;
    for (size_t i = 0; i < MultiHeadBilinear::K; ++i) {
        kan_input[i] = sigmoid(mh_scores[i]);
    }
    kan_input[4] = bilinear;
    kan_input[5] = dim_fraction;

    return kan_.evaluate(kan_input);
}

double ConceptModel::predict_refined(const FlexEmbedding& e, const FlexEmbedding& c) const {
    FlexEmbedding empty;
    return predict_refined(e, c, empty, empty);
}

// =============================================================================
// Training step (Adam optimizer) — identical to MicroModel
// =============================================================================

double ConceptModel::train_step(const FlexEmbedding& e, const FlexEmbedding& c,
                                 double target, const MicroTrainingConfig& config) {
    // Forward pass
    CoreVec v;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = b_[i];
        for (size_t j = 0; j < CORE_DIM; ++j) {
            sum += W_[i * CORE_DIM + j] * c.core[j];
        }
        v[i] = sum;
    }

    double z = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        z += e.core[i] * v[i];
    }

    double w = sigmoid(z);
    double error = w - target;
    double loss = 0.5 * error * error;
    double delta = error * w * (1.0 - w);

    state_.timestep += 1.0;
    double t = state_.timestep;

    double beta1 = config.adam_beta1;
    double beta2 = config.adam_beta2;
    double eps = config.adam_epsilon;
    double lr = config.learning_rate;

    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);

    for (size_t i = 0; i < CORE_DIM; ++i) {
        double grad_b = delta * e.core[i];

        state_.db_momentum[i] = beta1 * state_.db_momentum[i] + (1.0 - beta1) * grad_b;
        state_.db_variance[i] = beta2 * state_.db_variance[i] + (1.0 - beta2) * grad_b * grad_b;

        double m_hat = state_.db_momentum[i] / bc1;
        double v_hat = state_.db_variance[i] / bc2;

        b_[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

        for (size_t j = 0; j < CORE_DIM; ++j) {
            double grad_w = delta * e.core[i] * c.core[j];
            size_t idx = i * CORE_DIM + j;

            state_.dW_momentum[idx] = beta1 * state_.dW_momentum[idx] + (1.0 - beta1) * grad_w;
            state_.dW_variance[idx] = beta2 * state_.dW_variance[idx] + (1.0 - beta2) * grad_w * grad_w;

            double mw_hat = state_.dW_momentum[idx] / bc1;
            double vw_hat = state_.dW_variance[idx] / bc2;

            W_[idx] -= lr * mw_hat / (std::sqrt(vw_hat) + eps);
        }
    }

    state_.last_loss = loss;
    state_.total_samples += 1.0;
    if (loss < state_.best_loss) {
        state_.best_loss = loss;
    }

    return loss;
}

// =============================================================================
// Batch training
// =============================================================================

MicroTrainingResult ConceptModel::train(const std::vector<TrainingSample>& samples,
                                         const MicroTrainingConfig& config) {
    MicroTrainingResult result;

    if (samples.empty()) {
        result.converged = true;
        return result;
    }

    double prev_avg_loss = 1e9;

    for (size_t epoch = 0; epoch < config.max_epochs; ++epoch) {
        double total_loss = 0.0;

        for (const auto& sample : samples) {
            total_loss += train_step(sample.relation_embedding,
                                     sample.context_embedding,
                                     sample.target, config);
        }

        double avg_loss = total_loss / static_cast<double>(samples.size());
        result.epochs_run = epoch + 1;
        result.final_loss = avg_loss;

        double improvement = std::abs(prev_avg_loss - avg_loss);
        if (improvement < config.convergence_threshold && epoch > 0) {
            result.converged = true;
            break;
        }

        prev_avg_loss = avg_loss;
    }

    return result;
}

// =============================================================================
// Refined training (multi-head + KAN, numerical gradient)
// =============================================================================

void ConceptModel::train_refined(const FlexEmbedding& rel_emb, const FlexEmbedding& ctx_emb,
                                  const FlexEmbedding& concept_from,
                                  const FlexEmbedding& concept_to,
                                  double target, double learning_rate) {
    constexpr double eps = 1e-5;
    double base_output = predict_refined(rel_emb, ctx_emb, concept_from, concept_to);
    double base_loss = 0.5 * (base_output - target) * (base_output - target);

    // Train multi-head params (640)
    for (size_t i = 0; i < MultiHeadBilinear::TOTAL_PARAMS; ++i) {
        double orig = multihead_.params[i];
        multihead_.params[i] = orig + eps;
        double plus_output = predict_refined(rel_emb, ctx_emb, concept_from, concept_to);
        double plus_loss = 0.5 * (plus_output - target) * (plus_output - target);
        double grad = (plus_loss - base_loss) / eps;
        multihead_.params[i] = orig - learning_rate * grad;
    }

    // Train KAN params (280)
    for (size_t i = 0; i < FlexKAN::TOTAL_PARAMS; ++i) {
        double orig = kan_.params[i];
        kan_.params[i] = orig + eps;
        double plus_output = predict_refined(rel_emb, ctx_emb, concept_from, concept_to);
        double plus_loss = 0.5 * (plus_output - target) * (plus_output - target);
        double grad = (plus_loss - base_loss) / eps;
        kan_.params[i] = orig - learning_rate * grad;
    }
}

// =============================================================================
// Legacy KAN training
// =============================================================================

void ConceptModel::train_kan(double bilinear_score, double ctx_feature,
                              double validated_target, double learning_rate) {
    // Legacy: pass through to FlexKAN with only bilinear_score as meaningful input
    std::array<double, FlexKAN::INPUT_DIM> input{};
    input[4] = bilinear_score;
    input[5] = ctx_feature;
    kan_.train_step(input, validated_target, learning_rate);
}

// =============================================================================
// Serialization (1900 doubles)
// =============================================================================
// Layout: [0..939] bilinear, [940..1579] multihead, [1580..1859] KAN,
//         [1860..1874] patterns, [1875..1899] reserved

void ConceptModel::to_flat(std::array<double, CM_FLAT_SIZE>& out) const {
    size_t idx = 0;

    // W (256)
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) out[idx++] = W_[i];
    // b (16)
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = b_[i];
    // e_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = e_init_[i];
    // c_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = c_init_[i];

    // TrainingState
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) out[idx++] = state_.dW_momentum[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.db_momentum[i];
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) out[idx++] = state_.dW_variance[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.db_variance[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.e_grad_accum[i];
    for (size_t i = 0; i < CORE_DIM; ++i) out[idx++] = state_.c_grad_accum[i];
    out[idx++] = state_.timestep;
    out[idx++] = state_.last_loss;
    out[idx++] = state_.best_loss;
    out[idx++] = state_.total_samples;
    out[idx++] = state_.reserved_scalar;
    for (size_t i = 0; i < 55; ++i) out[idx++] = state_.reserved[i];

    // idx should be 940 here

    // MultiHeadBilinear (640)
    for (size_t i = 0; i < MultiHeadBilinear::TOTAL_PARAMS; ++i) out[idx++] = multihead_.params[i];

    // FlexKAN (280)
    for (size_t i = 0; i < FlexKAN::TOTAL_PARAMS; ++i) out[idx++] = kan_.params[i];

    // Pattern weights (15)
    out[idx++] = patterns_.shared_parent;
    out[idx++] = patterns_.transitive_causation;
    out[idx++] = patterns_.missing_link;
    out[idx++] = patterns_.weak_strengthening;
    out[idx++] = patterns_.contradictory_signal;
    out[idx++] = patterns_.chain_hypothesis;
    for (size_t i = 0; i < 9; ++i) out[idx++] = patterns_.reserved[i];

    // Reserved (25)
    for (size_t i = 0; i < 25; ++i) out[idx++] = reserved_[i];

    // idx should be 1900 here
}

void ConceptModel::from_flat(const std::array<double, CM_FLAT_SIZE>& in) {
    size_t idx = 0;

    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) W_[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) b_[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) e_init_[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) c_init_[i] = in[idx++];

    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) state_.dW_momentum[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.db_momentum[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) state_.dW_variance[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.db_variance[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.e_grad_accum[i] = in[idx++];
    for (size_t i = 0; i < CORE_DIM; ++i) state_.c_grad_accum[i] = in[idx++];
    state_.timestep = in[idx++];
    state_.last_loss = in[idx++];
    state_.best_loss = in[idx++];
    state_.total_samples = in[idx++];
    state_.reserved_scalar = in[idx++];
    for (size_t i = 0; i < 55; ++i) state_.reserved[i] = in[idx++];

    for (size_t i = 0; i < MultiHeadBilinear::TOTAL_PARAMS; ++i) multihead_.params[i] = in[idx++];

    for (size_t i = 0; i < FlexKAN::TOTAL_PARAMS; ++i) kan_.params[i] = in[idx++];

    patterns_.shared_parent = in[idx++];
    patterns_.transitive_causation = in[idx++];
    patterns_.missing_link = in[idx++];
    patterns_.weak_strengthening = in[idx++];
    patterns_.contradictory_signal = in[idx++];
    patterns_.chain_hypothesis = in[idx++];
    for (size_t i = 0; i < 9; ++i) patterns_.reserved[i] = in[idx++];

    for (size_t i = 0; i < 25; ++i) reserved_[i] = in[idx++];
}

} // namespace brain19
