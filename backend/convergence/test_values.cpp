// =============================================================================
// VALUE INSPECTION — Print actual outputs to verify correctness
// =============================================================================

#include "convergence_pipeline.hpp"
#include "convergence_kan.hpp"
#include "concept_router.hpp"
#include "concept_bank.hpp"
#include "gated_residual.hpp"
#include "relation_decoder.hpp"
#include "../core/relation_config.hpp"
#include "../cmodel/concept_model.hpp"
#include "../cursor/template_engine.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/stm.hpp"
#include "../memory/activation_level.hpp"
#include "../epistemic/epistemic_metadata.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace brain19;
using namespace brain19::convergence;

static void print_separator(const char* title) {
    std::cout << "\n══════════════════════════════════════════════════════\n";
    std::cout << "  " << title << "\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
}

// 1. ConvergencePort forward values
static void inspect_convergence_port() {
    print_separator("1. ConvergencePort Forward Values");

    ConceptModel cm;

    // Simulate convergence input: h(90) + k1_proj(32) = 122
    double input[122];
    for (size_t i = 0; i < 90; ++i) input[i] = 0.3 * std::sin(static_cast<double>(i) * 0.5);
    for (size_t i = 90; i < 122; ++i) input[i] = 0.1 * std::cos(static_cast<double>(i) * 0.3);

    double output[32];
    cm.forward_convergence(input, output);

    std::cout << "  Input (first 10 of 122): ";
    for (size_t i = 0; i < 10; ++i) std::cout << std::fixed << std::setprecision(4) << input[i] << " ";
    std::cout << "...\n";

    std::cout << "  Output (all 32):         ";
    for (size_t i = 0; i < 32; ++i) std::cout << std::fixed << std::setprecision(4) << output[i] << " ";
    std::cout << "\n";

    // Verify all in (-1,1) range
    double min_out = 1.0, max_out = -1.0, sum_out = 0.0;
    for (size_t i = 0; i < 32; ++i) {
        min_out = std::min(min_out, output[i]);
        max_out = std::max(max_out, output[i]);
        sum_out += output[i];
    }
    std::cout << "  Range: [" << min_out << ", " << max_out << "], Mean: " << sum_out / 32.0 << "\n";
    std::cout << "  OK: All outputs in tanh range (-1, 1): " << (min_out > -1.0 && max_out < 1.0 ? "YES" : "NO") << "\n";
}

// 2. Training convergence of ConvergencePort
static void inspect_convergence_training() {
    print_separator("2. ConvergencePort Training Convergence");

    ConceptModel cm;

    double input[122];
    for (size_t i = 0; i < 122; ++i) input[i] = 0.3 * std::sin(static_cast<double>(i) * 0.5);

    double target[32];
    for (size_t i = 0; i < 32; ++i) target[i] = 0.5;

    double output[32];

    for (int step = 0; step <= 200; step += 20) {
        cm.forward_convergence(input, output);
        double mse = 0.0;
        for (size_t i = 0; i < 32; ++i) {
            double d = output[i] - target[i];
            mse += d * d;
        }
        mse /= 32.0;
        std::cout << "  Step " << std::setw(3) << step << ": MSE=" << std::fixed << std::setprecision(6) << mse
                  << "  out[0]=" << std::setprecision(4) << output[0]
                  << "  out[15]=" << output[15]
                  << "  out[31]=" << output[31] << "\n";

        // Train for 20 steps
        if (step < 200) {
            for (int s = 0; s < 20; ++s) {
                cm.forward_convergence(input, output);
                double grad[32];
                for (size_t i = 0; i < 32; ++i) grad[i] = output[i] - target[i];
                cm.backward_convergence(input, output, grad, 0.005);
            }
        }
    }
}

// 3. Serialization round-trip bit-exact check
static void inspect_serialization() {
    print_separator("3. CM Serialization Round-Trip (V6)");

    ConceptModel cm1;
    // Set specific convergence port values
    cm1.convergence_port().W[0] = 0.123456789;
    cm1.convergence_port().W[100] = -0.987654321;
    cm1.convergence_port().b[0] = 0.42;
    cm1.convergence_port().b[31] = -0.13;

    std::array<double, CM_FLAT_SIZE> flat;
    cm1.to_flat(flat);

    ConceptModel cm2;
    cm2.from_flat(flat);

    std::cout << "  CM_FLAT_SIZE = " << CM_FLAT_SIZE << " (was " << CM_FLAT_SIZE_V5 << ")\n";
    std::cout << "  Memory per model: " << CM_FLAT_SIZE * 8 << " bytes ("
              << CM_FLAT_SIZE * 8 / 1024 << " KB)\n";
    std::cout << "  Convergence port at flat offsets [1900.." << CM_FLAT_SIZE - 1 << "]\n";
    std::cout << "  flat[1900] (W[0])   = " << std::setprecision(15) << flat[1900] << "\n";
    std::cout << "  flat[2000] (W[100]) = " << flat[2000] << "\n";
    std::cout << "  flat[5804] (b[0])   = " << flat[5804] << "\n";
    std::cout << "  flat[5835] (b[31])  = " << flat[5835] << "\n";

    // Check round-trip
    bool exact = true;
    exact &= (cm1.convergence_port().W[0] == cm2.convergence_port().W[0]);
    exact &= (cm1.convergence_port().W[100] == cm2.convergence_port().W[100]);
    exact &= (cm1.convergence_port().b[0] == cm2.convergence_port().b[0]);
    exact &= (cm1.convergence_port().b[31] == cm2.convergence_port().b[31]);

    // Check bilinear prediction preserved
    FlexEmbedding e, c;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e.core[i] = 0.5 * std::sin(static_cast<double>(i));
        c.core[i] = 0.3 * std::cos(static_cast<double>(i));
    }
    double pred1 = cm1.predict(e, c);
    double pred2 = cm2.predict(e, c);

    std::cout << "  Bilinear predict1 = " << pred1 << "\n";
    std::cout << "  Bilinear predict2 = " << pred2 << "\n";
    std::cout << "  Round-trip bit-exact: " << (exact ? "YES" : "NO") << "\n";
    std::cout << "  Bilinear preserved:   " << (pred1 == pred2 ? "YES" : "NO") << "\n";
}

// 4. V5 migration (zero convergence port)
static void inspect_migration() {
    print_separator("4. V5→V6 Migration (Zero Convergence Port)");

    ConceptModel cm_orig;
    std::array<double, CM_FLAT_SIZE> full_flat;
    cm_orig.to_flat(full_flat);

    // Simulate V5 format: only first 1900 doubles
    std::array<double, CM_FLAT_SIZE> v5_migrated{};
    for (size_t i = 0; i < CM_FLAT_SIZE_V5; ++i) v5_migrated[i] = full_flat[i];

    ConceptModel cm_migrated;
    cm_migrated.from_flat(v5_migrated);

    // Check all convergence port params are zero
    bool all_zero = true;
    for (size_t i = 0; i < ConvergencePort::W_SIZE; ++i) {
        if (cm_migrated.convergence_port().W[i] != 0.0) { all_zero = false; break; }
    }
    for (size_t i = 0; i < ConvergencePort::OUTPUT_DIM; ++i) {
        if (cm_migrated.convergence_port().b[i] != 0.0) { all_zero = false; break; }
    }

    // Forward through zero port should give all zeros
    double input[122];
    for (size_t i = 0; i < 122; ++i) input[i] = 1.0;
    double output[32];
    cm_migrated.forward_convergence(input, output);

    double max_out = 0.0;
    for (size_t i = 0; i < 32; ++i) max_out = std::max(max_out, std::abs(output[i]));

    std::cout << "  Convergence port all zeros: " << (all_zero ? "YES" : "NO") << "\n";
    std::cout << "  Max |output| with unit input: " << max_out << " (should be 0.0)\n";

    // Bilinear should still work
    FlexEmbedding e, c;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e.core[i] = 0.5; c.core[i] = 0.3;
    }
    double pred_orig = cm_orig.predict(e, c);
    double pred_migr = cm_migrated.predict(e, c);
    std::cout << "  Bilinear original:  " << pred_orig << "\n";
    std::cout << "  Bilinear migrated:  " << pred_migr << "\n";
    std::cout << "  Bilinear preserved: " << (pred_orig == pred_migr ? "YES" : "NO") << "\n";
}

// 5. RelationDecoder values
static void inspect_decoder() {
    print_separator("5. RelationDecoder Category Slices");

    RelationDecoder decoder;

    const char* cat_names[] = {
        "HIERARCHICAL", "COMPOSITIONAL", "CAUSAL", "OPPOSITION",
        "SIMILARITY", "TEMPORAL", "FUNCTIONAL", "EPISTEMIC", "CUSTOM"
    };
    RelationCategory cats[] = {
        RelationCategory::HIERARCHICAL, RelationCategory::COMPOSITIONAL,
        RelationCategory::CAUSAL, RelationCategory::OPPOSITION,
        RelationCategory::SIMILARITY, RelationCategory::TEMPORAL,
        RelationCategory::FUNCTIONAL, RelationCategory::EPISTEMIC,
        RelationCategory::CUSTOM_CATEGORY
    };

    for (size_t i = 0; i < 9; ++i) {
        auto s = decoder.get_slice(cats[i]);
        std::cout << "  " << std::setw(15) << std::left << cat_names[i]
                  << " dims [" << s.start << ".." << s.end - 1 << "] (" << s.size() << " dims)\n";
    }

    // Test with a sample logit vector
    std::vector<double> logits(32, 0.0);
    logits[8] = 2.0; logits[9] = 1.5; logits[10] = 1.0; logits[11] = 0.5; // CAUSAL
    logits[14] = 0.8; logits[15] = 0.7; // SIMILARITY

    auto activations = decoder.category_activations(logits);
    std::cout << "\n  Sample activations (CAUSAL=2.0/1.5/1.0/0.5, SIMILARITY=0.8/0.7):\n";
    for (const auto& [cat, score] : activations) {
        for (size_t i = 0; i < 9; ++i) {
            if (cats[i] == cat) {
                std::cout << "    " << std::setw(15) << std::left << cat_names[i]
                          << " = " << std::fixed << std::setprecision(4) << score << "\n";
            }
        }
    }

    auto dominant = decoder.dominant_category(logits);
    for (size_t i = 0; i < 9; ++i) {
        if (cats[i] == dominant) {
            std::cout << "  Dominant: " << cat_names[i] << "\n";
        }
    }
}

// 6. Epistemic framing
static void inspect_epistemic_framing() {
    print_separator("6. Epistemic Framing Examples");

    struct TestCase {
        float trust;
        EpistemicType type;
        const char* sentence;
    };

    TestCase cases[] = {
        {0.99f, EpistemicType::FACT,        "water is wet"},
        {0.90f, EpistemicType::THEORY,      "gravity is universal"},
        {0.70f, EpistemicType::THEORY,      "dark energy accelerates expansion"},
        {0.40f, EpistemicType::INFERENCE,    "the signal is from a pulsar"},
        {0.40f, EpistemicType::INFERENCE,    "the anomaly occurred yesterday"},
        {0.15f, EpistemicType::THEORY,       "wormholes connect galaxies"},
        {0.99f, EpistemicType::SPECULATION,  "aliens built the pyramids"},
        {0.99f, EpistemicType::HYPOTHESIS,   "dark matter decays slowly"},
    };

    for (const auto& tc : cases) {
        auto result = TemplateEngine::epistemic_frame(tc.trust, tc.type, tc.sentence);
        const char* type_str = "?";
        switch (tc.type) {
            case EpistemicType::FACT:        type_str = "FACT"; break;
            case EpistemicType::DEFINITION:  type_str = "DEFINITION"; break;
            case EpistemicType::THEORY:      type_str = "THEORY"; break;
            case EpistemicType::INFERENCE:   type_str = "INFERENCE"; break;
            case EpistemicType::HYPOTHESIS:  type_str = "HYPOTHESIS"; break;
            case EpistemicType::SPECULATION: type_str = "SPECULATION"; break;
        }
        std::cout << "  trust=" << std::fixed << std::setprecision(2) << tc.trust
                  << " type=" << std::setw(12) << std::left << type_str
                  << " -> \"" << result << "\"\n";
    }
}

// 7. STM Inhibition
static void inspect_stm_inhibition() {
    print_separator("7. STM Inhibition");

    ShortTermMemory stm;
    ContextId ctx = stm.create_context();

    stm.activate_concept(ctx, 1, 0.8, ActivationClass::CONTEXTUAL);
    stm.activate_concept(ctx, 2, 0.5, ActivationClass::CONTEXTUAL);
    stm.activate_concept(ctx, 3, 0.3, ActivationClass::CONTEXTUAL);

    std::cout << "  Before inhibition:\n";
    std::cout << "    concept 1: " << stm.get_concept_activation(ctx, 1) << "\n";
    std::cout << "    concept 2: " << stm.get_concept_activation(ctx, 2) << "\n";
    std::cout << "    concept 3: " << stm.get_concept_activation(ctx, 3) << "\n";

    stm.inhibit_concept(ctx, 1, 0.3);   // 0.8 - 0.3 = 0.5
    stm.inhibit_concept(ctx, 2, 0.7);   // 0.5 - 0.7 = clamp(0.0)
    stm.inhibit_concept(ctx, 3, 0.1);   // 0.3 - 0.1 = 0.2

    std::cout << "  After inhibition (0.3, 0.7, 0.1):\n";
    std::cout << "    concept 1: " << stm.get_concept_activation(ctx, 1)
              << " (expected 0.5)\n";
    std::cout << "    concept 2: " << stm.get_concept_activation(ctx, 2)
              << " (expected 0.0, clamped)\n";
    std::cout << "    concept 3: " << stm.get_concept_activation(ctx, 3)
              << " (expected 0.2)\n";
}

// 8. RelationBehavior config check
static void inspect_relation_behaviors() {
    print_separator("8. RelationBehavior Config");

    const char* cat_names[] = {
        "HIERARCHICAL", "COMPOSITIONAL", "CAUSAL", "SIMILARITY",
        "OPPOSITION", "EPISTEMIC", "TEMPORAL", "FUNCTIONAL", "CUSTOM"
    };
    RelationCategory cats[] = {
        RelationCategory::HIERARCHICAL, RelationCategory::COMPOSITIONAL,
        RelationCategory::CAUSAL, RelationCategory::SIMILARITY,
        RelationCategory::OPPOSITION, RelationCategory::EPISTEMIC,
        RelationCategory::TEMPORAL, RelationCategory::FUNCTIONAL,
        RelationCategory::CUSTOM_CATEGORY
    };

    std::cout << std::left << std::setw(15) << "  Category"
              << std::setw(8) << "SprdW"
              << std::setw(8) << "SprdD"
              << std::setw(8) << "EmbA"
              << std::setw(8) << "Decay"
              << std::setw(8) << "Inher"
              << "InherDir\n";
    std::cout << "  " << std::string(63, '-') << "\n";

    for (size_t i = 0; i < 9; ++i) {
        const auto& b = get_behavior(cats[i]);
        const char* dir = "NONE";
        switch (b.inherit_dir) {
            case InheritDirection::FORWARD: dir = "FWD"; break;
            case InheritDirection::REVERSE: dir = "REV"; break;
            case InheritDirection::BOTH:    dir = "BOTH"; break;
            default: break;
        }
        std::cout << "  " << std::setw(15) << cat_names[i]
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << b.spreading_weight
                  << std::setw(8) << b.spreading_direction
                  << std::setw(8) << b.embedding_alpha
                  << std::setw(8) << b.trust_decay_per_hop
                  << std::setw(8) << (b.inherit_properties ? "yes" : "no")
                  << dir << "\n";
    }
}

// 9. Full Convergence Pipeline values
static void inspect_pipeline() {
    print_separator("9. Full Convergence Pipeline Forward");

    ConvergencePipeline pipeline;

    // Create a random query
    std::vector<double> h(QUERY_DIM, 0.0);
    for (size_t i = 0; i < QUERY_DIM; ++i) {
        h[i] = 0.2 * std::sin(static_cast<double>(i) * 0.3);
    }

    auto result = pipeline.forward(h);

    std::cout << "  Query dim: " << QUERY_DIM << "\n";
    std::cout << "  Output dim: " << OUTPUT_DIM << "\n";
    std::cout << "  Agreement: " << std::fixed << std::setprecision(4) << result.agreement << "\n";
    std::cout << "  Ignition: " << (result.ignition == IgnitionMode::FAST ? "FAST" :
                                    result.ignition == IgnitionMode::DELIBERATE ? "DELIBERATE" : "CONFLICT") << "\n";
    std::cout << "  Routes: " << result.routes.size() << " active concepts\n";
    for (const auto& route : result.routes) {
        std::cout << "    concept " << route.concept_id << " weight=" << route.weight
                  << " raw_score=" << route.raw_score << "\n";
    }

    std::cout << "  Fused output (first 8): ";
    for (size_t i = 0; i < std::min(size_t(8), result.fused.size()); ++i) {
        std::cout << std::fixed << std::setprecision(4) << result.fused[i] << " ";
    }
    std::cout << "...\n";

    std::cout << "  G_out (first 8): ";
    for (size_t i = 0; i < std::min(size_t(8), result.G_out.size()); ++i) {
        std::cout << result.G_out[i] << " ";
    }
    std::cout << "...\n";

    std::cout << "  L_out (first 8): ";
    for (size_t i = 0; i < std::min(size_t(8), result.L_out.size()); ++i) {
        std::cout << result.L_out[i] << " ";
    }
    std::cout << "...\n";

    // Verify norms
    double fused_norm = 0.0, g_norm = 0.0, l_norm = 0.0;
    for (size_t i = 0; i < result.fused.size(); ++i) {
        fused_norm += result.fused[i] * result.fused[i];
        g_norm += result.G_out[i] * result.G_out[i];
        l_norm += result.L_out[i] * result.L_out[i];
    }
    std::cout << "  ||fused||=" << std::sqrt(fused_norm)
              << "  ||G||=" << std::sqrt(g_norm)
              << "  ||L||=" << std::sqrt(l_norm) << "\n";
}

// 10. Pipeline training convergence
static void inspect_pipeline_training() {
    print_separator("10. Pipeline Training (20 steps)");

    ConvergencePipeline pipeline;
    ConvergencePipeline::TrainingConfig tc;
    tc.lr_kan_l1 = 1e-4;
    tc.lr_kan_proj = 1e-4;
    tc.lr_kan_l2l3 = 1e-4;
    tc.lr_cm = 1e-4;
    tc.lr_router = 1e-4;
    tc.lr_gate = 1e-4;

    std::vector<double> h(QUERY_DIM, 0.0);
    for (size_t i = 0; i < QUERY_DIM; ++i) {
        h[i] = 0.2 * std::sin(static_cast<double>(i) * 0.3);
    }

    std::vector<double> target(OUTPUT_DIM, 0.5);

    for (int step = 0; step < 20; ++step) {
        double loss = pipeline.train_step(h, target, tc);
        auto result = pipeline.forward(h);
        std::cout << "  Step " << std::setw(2) << step
                  << ": loss=" << std::fixed << std::setprecision(6) << loss
                  << "  agreement=" << std::setprecision(4) << result.agreement
                  << "  ignition=" << (result.ignition == IgnitionMode::FAST ? "FAST" :
                                       result.ignition == IgnitionMode::DELIBERATE ? "DLIB" : "CONF")
                  << "\n";
    }
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Brain19 Convergence V2 — Value Inspection Report    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    inspect_convergence_port();
    inspect_convergence_training();
    inspect_serialization();
    inspect_migration();
    inspect_decoder();
    inspect_epistemic_framing();
    inspect_stm_inhibition();
    inspect_relation_behaviors();
    inspect_pipeline();
    inspect_pipeline_training();

    std::cout << "\n=== All value inspections complete ===\n";
    return 0;
}
