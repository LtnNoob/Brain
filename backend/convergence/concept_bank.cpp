#include "concept_bank.hpp"
#include <cmath>
#include <cassert>
#include <random>

namespace brain19 {
namespace convergence {

// ─── ConceptExpert ───────────────────────────────────────────────────────────

ConceptBank::ConceptExpert::ConceptExpert()
    : W(CM_OUTPUT_DIM * CM_INPUT_DIM)
    , b(CM_OUTPUT_DIM, 0.0)
{
    // Xavier init
    static std::mt19937 rng(777);
    double limit = std::sqrt(6.0 / (CM_INPUT_DIM + CM_OUTPUT_DIM));
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto& w : W) {
        w = dist(rng);
    }
}

std::vector<double> ConceptBank::ConceptExpert::forward(
    const std::vector<double>& input) const
{
    assert(input.size() == CM_INPUT_DIM);
    std::vector<double> out(CM_OUTPUT_DIM);

    for (size_t i = 0; i < CM_OUTPUT_DIM; ++i) {
        double z = b[i];
        for (size_t j = 0; j < CM_INPUT_DIM; ++j) {
            z += W[i * CM_INPUT_DIM + j] * input[j];
        }
        out[i] = std::tanh(z);
    }
    return out;
}

// ─── ConceptBank ─────────────────────────────────────────────────────────────

std::vector<double> ConceptBank::forward(
    const std::vector<double>& cm_input,
    const std::vector<ConceptId>& concept_ids,
    const std::vector<double>& weights)
{
    assert(cm_input.size() == CM_INPUT_DIM);
    assert(concept_ids.size() == weights.size());

    std::vector<double> L_out(CM_OUTPUT_DIM, 0.0);

    for (size_t k = 0; k < concept_ids.size(); ++k) {
        ensure_concept(concept_ids[k]);
        auto single = forward_single(cm_input, concept_ids[k]);

        for (size_t i = 0; i < CM_OUTPUT_DIM; ++i) {
            L_out[i] += weights[k] * single[i];
        }
    }

    return L_out;
}

std::vector<double> ConceptBank::forward_single(
    const std::vector<double>& cm_input,
    ConceptId concept_id)
{
    ensure_concept(concept_id);
    return models_.at(concept_id).forward(cm_input);
}

std::vector<double> ConceptBank::backward(
    const std::vector<double>& cm_input,
    const std::vector<ConceptId>& concept_ids,
    const std::vector<double>& weights,
    const std::vector<double>& d_L_out,
    double lr)
{
    assert(d_L_out.size() == CM_OUTPUT_DIM);

    std::vector<double> d_input(CM_INPUT_DIM, 0.0);

    for (size_t k = 0; k < concept_ids.size(); ++k) {
        auto it = models_.find(concept_ids[k]);
        if (it == models_.end()) continue;

        auto& expert = it->second;

        // Per-concept gradient: d_L_out * weight[k]
        // Forward was: out = tanh(W·input + b), L_out += w_k * out
        // d_out_i / d_z_i = 1 - tanh²(z_i)
        auto out = expert.forward(cm_input);

        for (size_t i = 0; i < CM_OUTPUT_DIM; ++i) {
            double d_out = d_L_out[i] * weights[k];
            double dtanh = 1.0 - out[i] * out[i];  // tanh derivative
            double d_z = d_out * dtanh;

            // Update weights
            for (size_t j = 0; j < CM_INPUT_DIM; ++j) {
                expert.W[i * CM_INPUT_DIM + j] -= lr * d_z * cm_input[j];
                d_input[j] += d_z * expert.W[i * CM_INPUT_DIM + j];
            }
            expert.b[i] -= lr * d_z;
        }
    }

    return d_input;
}

void ConceptBank::ensure_concept(ConceptId id) {
    if (models_.find(id) == models_.end()) {
        models_.emplace(id, ConceptExpert());
    }
}

void ConceptBank::ensure_concepts(const std::vector<ConceptId>& ids) {
    for (auto id : ids) {
        ensure_concept(id);
    }
}

bool ConceptBank::has_concept(ConceptId id) const {
    return models_.find(id) != models_.end();
}

} // namespace convergence
} // namespace brain19
