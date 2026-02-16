#include "convergence_pipeline.hpp"
#include <cmath>
#include <cassert>
#include <numeric>

namespace brain19 {
namespace convergence {

ConvergencePipeline::ConvergencePipeline() = default;

// ─── Forward ─────────────────────────────────────────────────────────────────

PipelineOutput ConvergencePipeline::forward(const std::vector<double>& h) {
    assert(h.size() == QUERY_DIM);

    PipelineOutput output;
    cache_.h = h;

    // Step 1: KAN Layer 1  (h → k1)
    cache_.k1 = kan_.forward_layer1(h);

    // Step 2: Router  (h → Top-K concepts + weights)
    // Can run in parallel with KAN L1 in a multi-threaded implementation
    output.routes = router_.route(h, ROUTER_TOP_K);
    router_.record_activation(output.routes);

    // Extract concept IDs and weights from routes
    cache_.concept_ids.clear();
    cache_.concept_weights.clear();
    for (const auto& r : output.routes) {
        cache_.concept_ids.push_back(r.concept_id);
        cache_.concept_weights.push_back(r.weight);
    }

    // Step 3: KAN Projection  (k1 → k1_proj for CM input)
    cache_.k1_proj = kan_.project_for_cm(cache_.k1);

    // Step 4: Build CM input  (h ⊕ k1_proj)
    cache_.cm_input.clear();
    cache_.cm_input.reserve(CM_INPUT_DIM);
    cache_.cm_input.insert(cache_.cm_input.end(), h.begin(), h.end());
    cache_.cm_input.insert(cache_.cm_input.end(), cache_.k1_proj.begin(), cache_.k1_proj.end());

    // Step 5: ConceptBank forward  (cm_input → L(h))
    cache_.L_out = concept_bank_.forward(cache_.cm_input, cache_.concept_ids, cache_.concept_weights);
    output.L_out = cache_.L_out;

    // Step 6: KAN Layers 2+3  (k1 ⊕ L(h) → G(h))
    // This is the CM-Feedback-Port: KAN Layer 2 receives CM output
    cache_.G_out = kan_.forward_layer2_3(cache_.k1, cache_.L_out);
    output.G_out = cache_.G_out;

    // Step 7: Gated Residual Convergence  (h, G(h), L(h) → fused)
    auto conv_result = gate_.converge(h, cache_.G_out, cache_.L_out);
    output.fused = conv_result.fused;
    output.agreement = conv_result.agreement;
    output.ignition = conv_result.mode;
    cache_.gate_values = conv_result.gate_values;

    return output;
}

// ─── Training ────────────────────────────────────────────────────────────────

double ConvergencePipeline::train_step(
    const std::vector<double>& h,
    const std::vector<double>& target)
{
    TrainingConfig config;
    return train_step(h, target, config);
}

double ConvergencePipeline::train_step(
    const std::vector<double>& h,
    const std::vector<double>& target,
    const TrainingConfig& config)
{
    assert(target.size() == OUTPUT_DIM);

    // Forward pass
    auto output = forward(h);

    // Compute MSE loss
    double loss = 0.0;
    std::vector<double> d_fused(OUTPUT_DIM);
    for (size_t i = 0; i < OUTPUT_DIM; ++i) {
        double diff = output.fused[i] - target[i];
        loss += diff * diff;
        d_fused[i] = 2.0 * diff / OUTPUT_DIM;  // d(MSE)/d(fused)
    }
    loss /= OUTPUT_DIM;

    // Backward through gate
    auto gate_grad = gate_.backward(cache_.h, cache_.G_out, cache_.L_out,
                                     cache_.gate_values, d_fused, config.lr_gate);

    // Backward through KAN Layers 2+3
    auto kan_grad = kan_.backward_layer2_3(gate_grad.d_G, config.lr_kan_l2l3, config.lr_kan_l2l3);

    // Backward through ConceptBank
    // d_L combines: gate's d_L + KAN L2's d_cm
    std::vector<double> d_L_total(CM_OUTPUT_DIM);
    for (size_t i = 0; i < CM_OUTPUT_DIM; ++i) {
        d_L_total[i] = gate_grad.d_L[i] + kan_grad.d_cm[i];
    }
    auto d_cm_input = concept_bank_.backward(cache_.cm_input, cache_.concept_ids,
                                              cache_.concept_weights, d_L_total, config.lr_cm);

    // d_cm_input = [d_h_part(90) | d_k1_proj(32)]
    // Backward through projection
    std::vector<double> d_k1_proj(d_cm_input.begin() + QUERY_DIM, d_cm_input.end());
    kan_.backward_projection(d_k1_proj, config.lr_kan_proj);

    // Backward through KAN Layer 1
    // d_k1 combines: KAN L2 backward's d_k1 + (projection backward would contribute, but we apply separately)
    kan_.backward_layer1(kan_grad.d_k1, config.lr_kan_l1);

    // Backward through router
    std::vector<double> d_weights(output.routes.size());
    // Approximate: d_weight_k ≈ d_L · single_cm_output_k (per-concept contribution)
    for (size_t k = 0; k < output.routes.size(); ++k) {
        auto single = concept_bank_.forward_single(cache_.cm_input, cache_.concept_ids[k]);
        double dw = 0.0;
        for (size_t i = 0; i < CM_OUTPUT_DIM; ++i) {
            dw += d_L_total[i] * single[i];
        }
        d_weights[k] = dw;
    }
    router_.backward(cache_.h, output.routes, d_weights, config.lr_router);

    return loss;
}

// ─── Accessors ───────────────────────────────────────────────────────────────

size_t ConvergencePipeline::total_params() const {
    return kan_.num_params()
         + gate_.num_params()
         + concept_bank_.num_concepts() * concept_bank_.params_per_concept()
         + router_.num_concepts() * ROUTER_DIM;
}

} // namespace convergence
} // namespace brain19
