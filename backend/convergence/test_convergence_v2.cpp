// =============================================================================
// CONVERGENCE V2 — Integration Tests for all new/modified components
// =============================================================================
// Tests: ConvergencePort, TemplateEngine EN, RelationDecoder, STM inhibition,
//        CM serialization round-trip, epistemic framing, relation-aware config

#include "convergence_config.hpp"
#include "relation_decoder.hpp"
#include "../core/relation_config.hpp"
#include "../cmodel/concept_model.hpp"
#include "../memory/stm.hpp"
#include "../memory/activation_level.hpp"
#include "../cursor/template_engine.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include "../memory/relation_type_registry.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cstring>

using namespace brain19;
using namespace brain19::convergence;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    void name(); \
    struct name##_reg { name##_reg() { std::cout << "  " #name "... "; std::cout.flush(); try { name(); tests_passed++; std::cout << "PASS\n"; } catch (const std::exception& e) { tests_failed++; std::cout << "FAIL: " << e.what() << "\n"; } catch (...) { tests_failed++; std::cout << "FAIL: unknown\n"; } } } name##_instance; \
    void name()

#define ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error( \
    std::string("Expected " #a " == " #b " but got ") + std::to_string(a) + " vs " + std::to_string(b)); } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error( \
    "Expected " #a " ≈ " #b " (diff=" + std::to_string(std::abs((a)-(b))) + ")"); } while(0)
#define ASSERT_TRUE(x) do { if (!(x)) throw std::runtime_error("Expected " #x " to be true"); } while(0)
#define ASSERT_FALSE(x) do { if ((x)) throw std::runtime_error("Expected " #x " to be false"); } while(0)
#define ASSERT_STR_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error( \
    std::string("Expected \"") + (b) + "\" but got \"" + (a) + "\""); } while(0)
#define ASSERT_STR_CONTAINS(haystack, needle) do { if ((haystack).find(needle) == std::string::npos) \
    throw std::runtime_error(std::string("Expected \"") + (haystack) + "\" to contain \"" + (needle) + "\""); } while(0)

// =============================================================================
// 1. ConvergencePort Tests
// =============================================================================

TEST(test_convergence_port_dimensions) {
    ConvergencePort port;
    port.initialize();
    ASSERT_EQ(ConvergencePort::INPUT_DIM, 122u);
    ASSERT_EQ(ConvergencePort::OUTPUT_DIM, 32u);
    ASSERT_EQ(ConvergencePort::W_SIZE, 32u * 122u);
    ASSERT_EQ(ConvergencePort::TOTAL_PARAMS, 32u * 122u + 32u);
}

TEST(test_convergence_port_forward_output_range) {
    ConvergencePort port;
    port.initialize();

    // Random-ish input
    double input[122];
    for (size_t i = 0; i < 122; ++i) {
        input[i] = std::sin(static_cast<double>(i) * 0.7) * 0.5;
    }

    double output[32];
    port.compute(input, output);

    // All outputs should be in (-1, 1) since tanh is the activation
    for (size_t i = 0; i < 32; ++i) {
        ASSERT_TRUE(output[i] > -1.0);
        ASSERT_TRUE(output[i] < 1.0);
    }
}

TEST(test_convergence_port_zero_input_zero_output) {
    ConvergencePort port;
    // Zero-initialized (migration case): W=0, b=0
    port.W.fill(0.0);
    port.b.fill(0.0);

    double input[122] = {};
    double output[32];
    port.compute(input, output);

    // tanh(0) = 0 for all outputs
    for (size_t i = 0; i < 32; ++i) {
        ASSERT_NEAR(output[i], 0.0, 1e-15);
    }
}

TEST(test_convergence_port_xavier_init_magnitude) {
    ConvergencePort port;
    port.initialize();

    // Xavier scale should be sqrt(2 / (122 + 32)) = sqrt(2/154) ≈ 0.114
    double expected_scale = std::sqrt(2.0 / 154.0);

    // Check that weights are in roughly the right range
    double max_w = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < ConvergencePort::W_SIZE; ++i) {
        double w = port.W[i];
        max_w = std::max(max_w, std::abs(w));
        sum_sq += w * w;
    }
    double rms = std::sqrt(sum_sq / ConvergencePort::W_SIZE);

    // Max weight should be <= scale (they're uniformly distributed in [-scale, scale])
    ASSERT_TRUE(max_w <= expected_scale * 1.01);
    // RMS should be roughly scale/sqrt(3) for uniform distribution ≈ 0.066
    ASSERT_TRUE(rms > 0.02);
    ASSERT_TRUE(rms < 0.15);

    // Biases should all be zero
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) {
        ASSERT_NEAR(port.b[i], 0.0, 1e-15);
    }
}

// =============================================================================
// 2. ConceptModel + ConvergencePort Integration
// =============================================================================

TEST(test_cm_forward_convergence) {
    ConceptModel cm;

    double input[122];
    for (size_t i = 0; i < 122; ++i) {
        input[i] = 0.1 * std::sin(static_cast<double>(i));
    }

    double output[32];
    cm.forward_convergence(input, output);

    // Should produce non-zero output (Xavier-initialized weights)
    double sum = 0.0;
    for (size_t i = 0; i < 32; ++i) {
        sum += std::abs(output[i]);
        ASSERT_TRUE(output[i] > -1.0 && output[i] < 1.0);
    }
    ASSERT_TRUE(sum > 0.01); // Not all zeros
}

TEST(test_cm_backward_convergence_updates_weights) {
    ConceptModel cm;

    double input[122];
    for (size_t i = 0; i < 122; ++i) {
        input[i] = 0.1 * std::cos(static_cast<double>(i));
    }

    // Forward
    double output[32];
    cm.forward_convergence(input, output);

    // Save original weight
    double w0_before = cm.convergence_port().W[0];
    double b0_before = cm.convergence_port().b[0];

    // Backward with non-zero gradient
    double grad[32];
    for (size_t i = 0; i < 32; ++i) grad[i] = 0.1;

    cm.backward_convergence(input, output, grad, 0.01);

    // Weights should have changed
    double w0_after = cm.convergence_port().W[0];
    double b0_after = cm.convergence_port().b[0];

    ASSERT_TRUE(w0_before != w0_after);
    ASSERT_TRUE(b0_before != b0_after);
}

TEST(test_cm_convergence_training_reduces_error) {
    ConceptModel cm;

    // Target: output should be all 0.5
    double target[32];
    for (size_t i = 0; i < 32; ++i) target[i] = 0.5;

    double input[122];
    for (size_t i = 0; i < 122; ++i) {
        input[i] = 0.3 * std::sin(static_cast<double>(i) * 0.5);
    }

    // Measure initial error
    double output[32];
    cm.forward_convergence(input, output);
    double initial_error = 0.0;
    for (size_t i = 0; i < 32; ++i) {
        double diff = output[i] - target[i];
        initial_error += diff * diff;
    }

    // Train for 100 steps
    for (int step = 0; step < 100; ++step) {
        cm.forward_convergence(input, output);
        double grad[32];
        for (size_t i = 0; i < 32; ++i) {
            grad[i] = output[i] - target[i]; // MSE gradient
        }
        cm.backward_convergence(input, output, grad, 0.005);
    }

    // Measure final error
    cm.forward_convergence(input, output);
    double final_error = 0.0;
    for (size_t i = 0; i < 32; ++i) {
        double diff = output[i] - target[i];
        final_error += diff * diff;
    }

    ASSERT_TRUE(final_error < initial_error);
}

// =============================================================================
// 3. CM Serialization Round-Trip (V6 with ConvergencePort)
// =============================================================================

TEST(test_cm_flat_size_v7) {
    // V7 layout: 5836 (V6) + 3904 (W_gate) + 32 (b_gate) = 9772
    ASSERT_EQ(CM_FLAT_SIZE, 9772u);
    ASSERT_EQ(CM_FLAT_SIZE_V6, 5836u);
    ASSERT_EQ(CM_FLAT_SIZE_V5, 1900u);
}

TEST(test_cm_serialization_roundtrip) {
    ConceptModel cm1;

    // Modify convergence port to non-default values
    for (size_t i = 0; i < ConvergencePort::W_SIZE; ++i) {
        cm1.convergence_port().W[i] = 0.42 * std::sin(static_cast<double>(i));
    }
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) {
        cm1.convergence_port().b[i] = 0.1 * static_cast<double>(i);
    }
    // Also modify gate weights
    for (size_t i = 0; i < ConvergencePort::W_GATE_SIZE; ++i) {
        cm1.convergence_port().W_gate[i] = 0.03 * std::cos(static_cast<double>(i));
    }
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) {
        cm1.convergence_port().b_gate[i] = -0.05 * static_cast<double>(i);
    }

    // Serialize
    std::array<double, CM_FLAT_SIZE> flat;
    cm1.to_flat(flat);

    // Deserialize into new model
    ConceptModel cm2;
    cm2.from_flat(flat);

    // Compare convergence port
    for (size_t i = 0; i < ConvergencePort::W_SIZE; ++i) {
        ASSERT_NEAR(cm1.convergence_port().W[i], cm2.convergence_port().W[i], 1e-15);
    }
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) {
        ASSERT_NEAR(cm1.convergence_port().b[i], cm2.convergence_port().b[i], 1e-15);
    }
    // Compare gate weights
    for (size_t i = 0; i < ConvergencePort::W_GATE_SIZE; ++i) {
        ASSERT_NEAR(cm1.convergence_port().W_gate[i], cm2.convergence_port().W_gate[i], 1e-15);
    }
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) {
        ASSERT_NEAR(cm1.convergence_port().b_gate[i], cm2.convergence_port().b_gate[i], 1e-15);
    }

    // Forward should produce same output
    double input[122];
    for (size_t i = 0; i < 122; ++i) input[i] = 0.1;
    double out1[32], out2[32];
    cm1.forward_convergence(input, out1);
    cm2.forward_convergence(input, out2);
    for (size_t i = 0; i < 32; ++i) {
        ASSERT_NEAR(out1[i], out2[i], 1e-15);
    }
}

TEST(test_cm_v5_migration_zero_convergence) {
    // Simulate V5 format: only first 1900 doubles have data, rest is zero
    ConceptModel cm_original;

    std::array<double, CM_FLAT_SIZE> flat{};
    // Fill only V5 portion
    std::array<double, CM_FLAT_SIZE> full_flat;
    cm_original.to_flat(full_flat);
    for (size_t i = 0; i < CM_FLAT_SIZE_V5; ++i) {
        flat[i] = full_flat[i];
    }
    // Convergence port + gate region [1900..9771] stays zero

    ConceptModel cm_migrated;
    cm_migrated.from_flat(flat);

    // With zero W, b, W_gate, b_gate:
    // gate = sigmoid(0) = 0.5
    // new_val = tanh(0) = 0
    // output = 0.5 * 0 + 0.5 * prev_state = 0.5 * prev_state
    // prev_state = input[90..121], which we set to 1.0
    double input[122];
    for (size_t i = 0; i < 122; ++i) input[i] = 1.0;
    double output[32];
    cm_migrated.forward_convergence(input, output);
    for (size_t i = 0; i < 32; ++i) {
        ASSERT_NEAR(output[i], 0.5, 1e-15);  // 0.5 * prev_state(1.0) = 0.5
    }
}

TEST(test_cm_bilinear_preserved_in_v6) {
    // Verify the bilinear path still works after V6 changes
    ConceptModel cm;
    FlexEmbedding e, c;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e.core[i] = 0.5 * std::sin(static_cast<double>(i));
        c.core[i] = 0.3 * std::cos(static_cast<double>(i));
    }

    double pred = cm.predict(e, c);
    ASSERT_TRUE(pred > 0.0 && pred < 1.0); // sigmoid output

    // Round-trip should preserve prediction
    std::array<double, CM_FLAT_SIZE> flat;
    cm.to_flat(flat);
    ConceptModel cm2;
    cm2.from_flat(flat);
    double pred2 = cm2.predict(e, c);
    ASSERT_NEAR(pred, pred2, 1e-15);
}

// =============================================================================
// 4. RelationDecoder Tests
// =============================================================================

TEST(test_relation_decoder_slice_coverage) {
    RelationDecoder decoder;

    // All 9 categories should be mapped
    auto s_hier = decoder.get_slice(RelationCategory::HIERARCHICAL);
    auto s_comp = decoder.get_slice(RelationCategory::COMPOSITIONAL);
    auto s_caus = decoder.get_slice(RelationCategory::CAUSAL);
    auto s_oppo = decoder.get_slice(RelationCategory::OPPOSITION);
    auto s_simi = decoder.get_slice(RelationCategory::SIMILARITY);
    auto s_temp = decoder.get_slice(RelationCategory::TEMPORAL);
    auto s_func = decoder.get_slice(RelationCategory::FUNCTIONAL);
    auto s_epis = decoder.get_slice(RelationCategory::EPISTEMIC);
    auto s_cust = decoder.get_slice(RelationCategory::CUSTOM_CATEGORY);

    // Check sizes
    ASSERT_EQ(s_hier.size(), 4u);
    ASSERT_EQ(s_comp.size(), 4u);
    ASSERT_EQ(s_caus.size(), 4u);
    ASSERT_EQ(s_oppo.size(), 2u);
    ASSERT_EQ(s_simi.size(), 6u);
    ASSERT_EQ(s_temp.size(), 4u);
    ASSERT_EQ(s_func.size(), 4u);
    ASSERT_EQ(s_epis.size(), 2u);
    ASSERT_EQ(s_cust.size(), 2u);

    // Total should be 32 (OUTPUT_DIM)
    size_t total = s_hier.size() + s_comp.size() + s_caus.size() +
                   s_oppo.size() + s_simi.size() + s_temp.size() +
                   s_func.size() + s_epis.size() + s_cust.size();
    ASSERT_EQ(total, 32u);

    // No gaps: end of one == start of next
    ASSERT_EQ(s_hier.end, s_comp.start);
    ASSERT_EQ(s_comp.end, s_caus.start);
    ASSERT_EQ(s_caus.end, s_oppo.start);
    ASSERT_EQ(s_oppo.end, s_simi.start);
    ASSERT_EQ(s_simi.end, s_temp.start);
    ASSERT_EQ(s_temp.end, s_func.start);
    ASSERT_EQ(s_func.end, s_epis.start);
    ASSERT_EQ(s_epis.end, s_cust.start);
    ASSERT_EQ(s_cust.end, 32u);
}

TEST(test_relation_decoder_decode_scalar) {
    RelationDecoder decoder;

    // Create logits where CAUSAL dims (8-11) are all 1.0, rest 0.0
    std::vector<double> logits(32, 0.0);
    logits[8] = 1.0; logits[9] = 1.0; logits[10] = 1.0; logits[11] = 1.0;

    double causal_scalar = decoder.decode_scalar(logits, RelationCategory::CAUSAL);
    ASSERT_NEAR(causal_scalar, 1.0, 1e-10);

    double hier_scalar = decoder.decode_scalar(logits, RelationCategory::HIERARCHICAL);
    ASSERT_NEAR(hier_scalar, 0.0, 1e-10);
}

TEST(test_relation_decoder_dominant_category) {
    RelationDecoder decoder;

    // SIMILARITY dims (14-19) are the most active
    std::vector<double> logits(32, 0.1);
    for (size_t i = 14; i < 20; ++i) logits[i] = 5.0;

    auto dominant = decoder.dominant_category(logits);
    ASSERT_TRUE(dominant == RelationCategory::SIMILARITY);
}

TEST(test_relation_decoder_category_activations_sorted) {
    RelationDecoder decoder;

    std::vector<double> logits(32, 0.0);
    // TEMPORAL strongest, then CAUSAL, rest zero
    logits[20] = 3.0; logits[21] = 3.0; logits[22] = 3.0; logits[23] = 3.0;
    logits[8] = 1.0; logits[9] = 1.0; logits[10] = 1.0; logits[11] = 1.0;

    auto activations = decoder.category_activations(logits);

    // First should be TEMPORAL (mean abs = 3.0)
    ASSERT_TRUE(activations[0].first == RelationCategory::TEMPORAL);
    ASSERT_NEAR(activations[0].second, 3.0, 1e-10);

    // Second should be CAUSAL (mean abs = 1.0)
    ASSERT_TRUE(activations[1].first == RelationCategory::CAUSAL);
    ASSERT_NEAR(activations[1].second, 1.0, 1e-10);

    // Rest should be 0
    ASSERT_NEAR(activations[2].second, 0.0, 1e-10);
}

// =============================================================================
// 5. STM Inhibition Tests
// =============================================================================

TEST(test_stm_inhibit_concept_reduces_activation) {
    ShortTermMemory stm;
    ContextId ctx = stm.create_context();
    ConceptId cid = 42;

    stm.activate_concept(ctx, cid, 0.8, ActivationClass::CONTEXTUAL);
    double before = stm.get_concept_activation(ctx, cid);
    ASSERT_NEAR(before, 0.8, 1e-10);

    stm.inhibit_concept(ctx, cid, 0.3);
    double after = stm.get_concept_activation(ctx, cid);
    ASSERT_NEAR(after, 0.5, 1e-10);
}

TEST(test_stm_inhibit_concept_clamps_at_zero) {
    ShortTermMemory stm;
    ContextId ctx = stm.create_context();
    ConceptId cid = 99;

    stm.activate_concept(ctx, cid, 0.2, ActivationClass::CONTEXTUAL);
    stm.inhibit_concept(ctx, cid, 0.5); // More than current activation
    double after = stm.get_concept_activation(ctx, cid);
    ASSERT_NEAR(after, 0.0, 1e-10);
}

TEST(test_stm_inhibit_nonexistent_concept_noop) {
    ShortTermMemory stm;
    ContextId ctx = stm.create_context();
    ConceptId cid = 999;

    // Should not crash — concept not in STM
    stm.inhibit_concept(ctx, cid, 0.5);
    double after = stm.get_concept_activation(ctx, cid);
    ASSERT_NEAR(after, 0.0, 1e-10);
}

// =============================================================================
// 6. Epistemic Framing Tests
// =============================================================================

TEST(test_epistemic_frame_high_trust_passthrough) {
    // trust >= 0.95 → sentence unchanged
    auto result = TemplateEngine::epistemic_frame(0.99f, EpistemicType::FACT, "water is wet");
    ASSERT_STR_EQ(result, "water is wet");
}

TEST(test_epistemic_frame_generally) {
    // trust >= 0.85 → "generally, ..."
    auto result = TemplateEngine::epistemic_frame(0.90f, EpistemicType::THEORY, "cats are mammals");
    ASSERT_STR_EQ(result, "generally, cats are mammals");
}

TEST(test_epistemic_frame_likely) {
    // trust >= 0.60 → "likely, ..."
    auto result = TemplateEngine::epistemic_frame(0.70f, EpistemicType::THEORY, "it will rain");
    ASSERT_STR_EQ(result, "likely, it will rain");
}

TEST(test_epistemic_frame_might_be) {
    // trust >= 0.30, has " is " → replace with " might be "
    auto result = TemplateEngine::epistemic_frame(0.40f, EpistemicType::INFERENCE, "the cat is hungry");
    ASSERT_STR_CONTAINS(result, "might be");
}

TEST(test_epistemic_frame_might_be_no_is) {
    // trust >= 0.30, no " is " → "possibly, ..."
    auto result = TemplateEngine::epistemic_frame(0.40f, EpistemicType::INFERENCE, "the cat ate food");
    ASSERT_STR_EQ(result, "possibly, the cat ate food");
}

TEST(test_epistemic_frame_speculation_override) {
    // SPECULATION overrides trust
    auto result = TemplateEngine::epistemic_frame(0.99f, EpistemicType::SPECULATION, "aliens exist");
    ASSERT_STR_EQ(result, "it's speculated that aliens exist");
}

TEST(test_epistemic_frame_hypothesis_override) {
    // HYPOTHESIS overrides trust
    auto result = TemplateEngine::epistemic_frame(0.99f, EpistemicType::HYPOTHESIS, "dark matter decays");
    ASSERT_STR_EQ(result, "it's hypothesized that dark matter decays");
}

TEST(test_epistemic_frame_very_low_trust) {
    // trust < 0.30 → "speculated"
    auto result = TemplateEngine::epistemic_frame(0.10f, EpistemicType::THEORY, "wormholes connect");
    ASSERT_STR_EQ(result, "it's speculated that wormholes connect");
}

// =============================================================================
// 7. English Template Tests
// =============================================================================

TEST(test_english_template_is_a) {
    auto tmpl = TemplateEngine::relation_template_en(RelationType::IS_A);
    ASSERT_STR_CONTAINS(tmpl, "{subject}");
    ASSERT_STR_CONTAINS(tmpl, "{object}");
    ASSERT_STR_CONTAINS(tmpl, "is a");
}

TEST(test_english_template_causes) {
    auto tmpl = TemplateEngine::relation_template_en(RelationType::CAUSES);
    ASSERT_STR_CONTAINS(tmpl, "causes");
}

TEST(test_english_template_contradicts) {
    auto tmpl = TemplateEngine::relation_template_en(RelationType::CONTRADICTS);
    ASSERT_STR_CONTAINS(tmpl, "contradicts");
}

TEST(test_english_template_all_builtins_have_templates) {
    // All 20 builtin RelationTypes should produce non-fallback templates
    auto& reg = RelationTypeRegistry::instance();
    auto all = reg.builtin_types();
    for (auto rt : all) {
        auto tmpl = TemplateEngine::relation_template_en(rt);
        ASSERT_TRUE(!tmpl.empty());
        ASSERT_STR_CONTAINS(tmpl, "{subject}");
        ASSERT_STR_CONTAINS(tmpl, "{object}");
    }
}

// =============================================================================
// 8. RelationBehavior Config Tests (integration)
// =============================================================================

TEST(test_relation_behavior_opposition_is_inhibitory) {
    const auto& beh = get_behavior(RelationCategory::OPPOSITION);
    ASSERT_TRUE(beh.spreading_direction < 0.0);
    ASSERT_TRUE(beh.embedding_alpha < 0.0);
}

TEST(test_relation_behavior_compositional_reverse_inherit) {
    const auto& beh = get_behavior(RelationCategory::COMPOSITIONAL);
    ASSERT_TRUE(beh.inherit_properties);
    ASSERT_TRUE(beh.inherit_dir == InheritDirection::REVERSE);
    ASSERT_NEAR(beh.trust_decay_per_hop, 1.0, 1e-10); // structural facts don't decay
}

TEST(test_relation_behavior_causal_forward_inherit) {
    const auto& beh = get_behavior(RelationCategory::CAUSAL);
    ASSERT_TRUE(beh.inherit_dir == InheritDirection::FORWARD);
    ASSERT_TRUE(beh.trust_decay_per_hop < 1.0); // causal chains decay
}

TEST(test_relation_behavior_type_lookup) {
    // Verify that RelationType → RelationCategory → RelationBehavior chain works
    const auto& beh_causes = get_behavior(RelationType::CAUSES);
    const auto& beh_causal = get_behavior(RelationCategory::CAUSAL);
    ASSERT_NEAR(beh_causes.spreading_weight, beh_causal.spreading_weight, 1e-15);
    ASSERT_NEAR(beh_causes.embedding_alpha, beh_causal.embedding_alpha, 1e-15);
}

// =============================================================================
// 9. ConvergencePort Gradient Correctness (numerical check)
// =============================================================================

TEST(test_convergence_port_gradient_numerical) {
    ConceptModel cm;

    double input[122];
    for (size_t i = 0; i < 122; ++i) {
        input[i] = 0.2 * std::sin(static_cast<double>(i) * 0.3);
    }

    // Forward
    double output[32];
    cm.forward_convergence(input, output);

    // Compute numerical gradient for W[0] using central difference
    constexpr double eps = 1e-5;
    double w0_orig = cm.convergence_port().W[0];

    // f(w + eps)
    cm.convergence_port().W[0] = w0_orig + eps;
    double out_plus[32];
    cm.forward_convergence(input, out_plus);

    // f(w - eps)
    cm.convergence_port().W[0] = w0_orig - eps;
    double out_minus[32];
    cm.forward_convergence(input, out_minus);

    cm.convergence_port().W[0] = w0_orig; // restore

    // Numerical gradient of output[0] w.r.t. W[0]
    // W[0] is W[0][0], which affects output[0] via:
    //   output[0] = tanh(sum_j W[0][j]*input[j] + b[0])
    // d output[0] / d W[0][0] = (1 - tanh²) * input[0]
    double numerical_grad_out0 = (out_plus[0] - out_minus[0]) / (2.0 * eps);

    // Analytical gradient: d tanh(z)/dW[0][0] = (1-y²) * input[0]
    double y = output[0];
    double analytical_grad_out0 = (1.0 - y * y) * input[0];

    ASSERT_NEAR(numerical_grad_out0, analytical_grad_out0, 1e-5);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== Brain19 Convergence V2 Integration Tests ===\n\n";
    // Tests are auto-registered and run before main()
    std::cout << "\n" << tests_passed << " passed, " << tests_failed << " failed\n";
    return tests_failed > 0 ? 1 : 0;
}
