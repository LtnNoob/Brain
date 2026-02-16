#include "convergence_pipeline.hpp"
#include "convergence_kan.hpp"
#include "concept_router.hpp"
#include "concept_bank.hpp"
#include "gated_residual.hpp"
#include "../core/relation_config.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

using namespace brain19;
using namespace brain19::convergence;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    void name(); \
    struct name##_reg { name##_reg() { std::cout << "  " #name "... "; try { name(); tests_passed++; std::cout << "PASS\n"; } catch (const std::exception& e) { tests_failed++; std::cout << "FAIL: " << e.what() << "\n"; } catch (...) { tests_failed++; std::cout << "FAIL: unknown\n"; } } } name##_instance; \
    void name()

#define ASSERT_EQ(a, b) do { if ((a) != (b)) throw std::runtime_error("Expected " #a " == " #b); } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("Expected " #a " ≈ " #b " (diff=" + std::to_string(std::abs((a)-(b))) + ")"); } while(0)
#define ASSERT_TRUE(x) do { if (!(x)) throw std::runtime_error("Expected " #x " to be true"); } while(0)

// ─── RelationConfig Tests ────────────────────────────────────────────────────

TEST(test_relation_config_all_categories) {
    // All 9 categories must have a behavior entry
    const auto& behaviors = get_relation_behaviors();
    ASSERT_TRUE(behaviors.count(RelationCategory::HIERARCHICAL));
    ASSERT_TRUE(behaviors.count(RelationCategory::COMPOSITIONAL));
    ASSERT_TRUE(behaviors.count(RelationCategory::CAUSAL));
    ASSERT_TRUE(behaviors.count(RelationCategory::SIMILARITY));
    ASSERT_TRUE(behaviors.count(RelationCategory::OPPOSITION));
    ASSERT_TRUE(behaviors.count(RelationCategory::EPISTEMIC));
    ASSERT_TRUE(behaviors.count(RelationCategory::TEMPORAL));
    ASSERT_TRUE(behaviors.count(RelationCategory::FUNCTIONAL));
    ASSERT_TRUE(behaviors.count(RelationCategory::CUSTOM_CATEGORY));
    ASSERT_EQ(behaviors.size(), 9u);
}

TEST(test_relation_config_opposition_inhibitory) {
    const auto& b = get_behavior(RelationCategory::OPPOSITION);
    ASSERT_TRUE(b.spreading_direction < 0);
    ASSERT_TRUE(b.embedding_alpha < 0);
    ASSERT_TRUE(!b.inherit_properties);
}

TEST(test_relation_config_compositional_reverse) {
    const auto& b = get_behavior(RelationCategory::COMPOSITIONAL);
    ASSERT_EQ(static_cast<int>(b.inherit_dir), static_cast<int>(InheritDirection::REVERSE));
    ASSERT_NEAR(b.trust_decay_per_hop, 1.0f, 0.01f);
}

TEST(test_relation_config_fallback) {
    // Unknown category should return default
    auto cat = static_cast<RelationCategory>(99);
    const auto& b = get_behavior(cat);
    ASSERT_NEAR(b.spreading_weight, 0.5f, 0.01f);
}

// ─── ConvergenceKAN Tests ────────────────────────────────────────────────────

TEST(test_kan_forward_dimensions) {
    ConvergenceKAN kan;

    std::vector<double> h(QUERY_DIM, 0.1);
    auto k1 = kan.forward_layer1(h);
    ASSERT_EQ(k1.size(), KAN_L1_OUT);

    auto k1_proj = kan.project_for_cm(k1);
    ASSERT_EQ(k1_proj.size(), KAN_PROJ_OUT);

    std::vector<double> cm_out(CM_OUTPUT_DIM, 0.5);
    auto g_out = kan.forward_layer2_3(k1, cm_out);
    ASSERT_EQ(g_out.size(), KAN_L3_OUT);
}

TEST(test_kan_num_params) {
    ConvergenceKAN kan;
    size_t params = kan.num_params();
    // Should be substantial (3 KAN layers + projection)
    ASSERT_TRUE(params > 10000);
}

// ─── CentroidRouter Tests ───────────────────────────────────────────────────

TEST(test_router_basic) {
    CentroidRouter router;

    // Add some concepts
    for (ConceptId id = 1; id <= 10; ++id) {
        std::vector<double> centroid(ROUTER_DIM);
        centroid[id - 1] = 1.0;  // One-hot-ish
        router.set_centroid(id, centroid);
    }

    ASSERT_EQ(router.num_concepts(), 10u);

    // Route a query that aligns with concept 3
    std::vector<double> h(ROUTER_DIM, 0.0);
    h[2] = 1.0;
    auto routes = router.route(h, 4);

    ASSERT_EQ(routes.size(), 4u);
    ASSERT_EQ(routes[0].concept_id, 3u);  // Best match
}

TEST(test_router_weights_sum_to_one) {
    CentroidRouter router;

    for (ConceptId id = 1; id <= 5; ++id) {
        std::vector<double> centroid(ROUTER_DIM, 0.0);
        centroid[0] = static_cast<double>(id);
        router.set_centroid(id, centroid);
    }

    std::vector<double> h(ROUTER_DIM, 0.0);
    h[0] = 1.0;
    auto routes = router.route(h, 3);

    double sum = 0.0;
    for (const auto& r : routes) sum += r.weight;
    ASSERT_NEAR(sum, 1.0, 1e-6);
}

// ─── GatedResidualPoE Tests ─────────────────────────────────────────────────

TEST(test_gate_agreement_identical) {
    std::vector<double> G(OUTPUT_DIM, 0.5);
    std::vector<double> L(OUTPUT_DIM, 0.5);
    float agreement = GatedResidualPoE::compute_agreement(G, L);
    ASSERT_NEAR(agreement, 1.0f, 0.01f);
}

TEST(test_gate_agreement_opposite) {
    std::vector<double> G(OUTPUT_DIM, 1.0);
    std::vector<double> L(OUTPUT_DIM, -1.0);
    float agreement = GatedResidualPoE::compute_agreement(G, L);
    ASSERT_TRUE(agreement < 0.1f);
}

TEST(test_gate_ignition_fast) {
    ASSERT_EQ(static_cast<int>(GatedResidualPoE::check_ignition(0.90f)),
              static_cast<int>(IgnitionMode::FAST));
}

TEST(test_gate_ignition_deliberate) {
    ASSERT_EQ(static_cast<int>(GatedResidualPoE::check_ignition(0.60f)),
              static_cast<int>(IgnitionMode::DELIBERATE));
}

TEST(test_gate_ignition_conflict) {
    ASSERT_EQ(static_cast<int>(GatedResidualPoE::check_ignition(0.20f)),
              static_cast<int>(IgnitionMode::CONFLICT));
}

TEST(test_gate_convergence_output_dim) {
    GatedResidualPoE gate;
    std::vector<double> h(QUERY_DIM, 0.1);
    std::vector<double> G(OUTPUT_DIM, 0.5);
    std::vector<double> L(OUTPUT_DIM, 0.3);

    auto result = gate.converge(h, G, L);
    ASSERT_EQ(result.fused.size(), OUTPUT_DIM);
}

TEST(test_gate_fast_returns_G) {
    GatedResidualPoE gate;
    std::vector<double> h(QUERY_DIM, 0.1);
    std::vector<double> G(OUTPUT_DIM, 0.5);
    std::vector<double> L = G;  // Identical → FAST

    auto result = gate.converge(h, G, L);
    ASSERT_EQ(static_cast<int>(result.mode), static_cast<int>(IgnitionMode::FAST));
    for (size_t i = 0; i < OUTPUT_DIM; ++i) {
        ASSERT_NEAR(result.fused[i], G[i], 1e-10);
    }
}

// ─── ConceptBank Tests ──────────────────────────────────────────────────────

TEST(test_concept_bank_forward) {
    ConceptBank bank;

    std::vector<double> cm_input(CM_INPUT_DIM, 0.1);
    std::vector<ConceptId> ids = {1, 2, 3, 4};
    std::vector<double> weights = {0.4, 0.3, 0.2, 0.1};

    auto L = bank.forward(cm_input, ids, weights);
    ASSERT_EQ(L.size(), CM_OUTPUT_DIM);

    // Output should be bounded by tanh
    for (double v : L) {
        ASSERT_TRUE(v >= -1.0 && v <= 1.0);
    }
}

TEST(test_concept_bank_creates_on_demand) {
    ConceptBank bank;
    ASSERT_EQ(bank.num_concepts(), 0u);

    std::vector<double> cm_input(CM_INPUT_DIM, 0.1);
    bank.forward_single(cm_input, 42);
    ASSERT_TRUE(bank.has_concept(42));
    ASSERT_EQ(bank.num_concepts(), 1u);
}

// ─── ConvergencePipeline Tests ──────────────────────────────────────────────

TEST(test_pipeline_forward) {
    ConvergencePipeline pipeline;

    // Need at least some concepts for routing
    for (ConceptId id = 1; id <= 20; ++id) {
        std::vector<double> centroid(ROUTER_DIM, 0.0);
        centroid[id % ROUTER_DIM] = 1.0;
        pipeline.router().set_centroid(id, centroid);
    }

    std::vector<double> h(QUERY_DIM, 0.1);
    auto output = pipeline.forward(h);

    ASSERT_EQ(output.fused.size(), OUTPUT_DIM);
    ASSERT_EQ(output.G_out.size(), OUTPUT_DIM);
    ASSERT_EQ(output.L_out.size(), OUTPUT_DIM);
    ASSERT_TRUE(output.agreement >= 0.0f && output.agreement <= 1.0f);
    ASSERT_EQ(output.routes.size(), ROUTER_TOP_K);
}

TEST(test_pipeline_train_reduces_loss) {
    ConvergencePipeline pipeline;

    // Set up concepts
    for (ConceptId id = 1; id <= 20; ++id) {
        std::vector<double> centroid(ROUTER_DIM, 0.0);
        centroid[id % ROUTER_DIM] = 0.5;
        pipeline.router().set_centroid(id, centroid);
    }

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);

    std::vector<double> h(QUERY_DIM);
    std::vector<double> target(OUTPUT_DIM);
    for (auto& v : h) v = dist(rng);
    for (auto& v : target) v = dist(rng);

    double first_loss = pipeline.train_step(h, target);
    double last_loss = first_loss;

    // Train for a few steps
    for (int i = 0; i < 20; ++i) {
        last_loss = pipeline.train_step(h, target);
    }

    // Loss should decrease (on a single example it should overfit)
    ASSERT_TRUE(last_loss < first_loss);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n=== Brain19 Convergence Tests ===\n\n";
    // Tests are auto-registered by TEST() macro constructors
    std::cout << "\n" << tests_passed << " passed, " << tests_failed << " failed\n";
    return tests_failed > 0 ? 1 : 0;
}
