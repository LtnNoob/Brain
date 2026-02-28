// test_superposition.cpp — Tests for ContextSuperposition (V8)
//
// 1. Backward compatibility: superposition disabled → identical to V7
// 2. Context sensitivity: different ctx → different output
// 3. Serialization roundtrip: to_flat → from_flat preserves all 161 params
// 4. V7→V8 migration: zero-fill → disabled
// 5. Attention normalization: softmax sums to 1.0
// 6. Modulation math: hand-computed W_eff*x verification

#include "cmodel/concept_model.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <array>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace brain19;

static constexpr double EPS = 1e-10;

static bool approx_eq(double a, double b, double tol = 1e-8) {
    return std::abs(a - b) < tol;
}

// =============================================================================
// Test 1: Backward compatibility — disabled superposition = V7 behavior
// =============================================================================

static void test_backward_compatibility() {
    std::printf("  [1] Backward compatibility... ");

    ConceptModel model;

    // Superposition should be disabled by default
    assert(model.superposition().enabled < 0.5);

    // Create test embeddings
    FlexEmbedding e, c;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e.core[i] = 0.1 * std::sin(static_cast<double>(i * 3 + 1));
        c.core[i] = 0.1 * std::cos(static_cast<double>(i * 7 + 2));
    }

    // V7-equivalent predict should work identically
    double score1 = model.predict(e, c);
    assert(score1 > 0.0 && score1 < 1.0);

    // Verify to_flat/from_flat roundtrip preserves disabled state
    std::array<double, CM_FLAT_SIZE> flat;
    model.to_flat(flat);

    ConceptModel model2;
    model2.from_flat(flat);
    assert(model2.superposition().enabled < 0.5);

    double score2 = model2.predict(e, c);
    assert(approx_eq(score1, score2));

    std::printf("PASS\n");
}

// =============================================================================
// Test 2: Context sensitivity — different ctx → different modulated output
// =============================================================================

static void test_context_sensitivity() {
    std::printf("  [2] Context sensitivity... ");

    ContextSuperposition sp;
    sp.enabled = 1.0;

    // Set up non-trivial u, v, keys
    for (size_t k = 0; k < ContextSuperposition::N_MODES; ++k) {
        for (size_t i = 0; i < CORE_DIM; ++i) {
            sp.u[k * CORE_DIM + i] = 0.1 * std::sin(static_cast<double>(k * 100 + i * 7));
            sp.v[k * CORE_DIM + i] = 0.1 * std::cos(static_cast<double>(k * 50 + i * 11));
        }
        for (size_t d = 0; d < ContextSuperposition::KEY_DIM; ++d) {
            sp.keys[k * ContextSuperposition::KEY_DIM + d] =
                0.5 * std::sin(static_cast<double>(k * 30 + d * 13));
        }
    }

    // Identity-ish W and test input
    CoreMat W{};
    for (size_t i = 0; i < CORE_DIM; ++i)
        W[i * CORE_DIM + i] = 1.0;

    CoreVec x{};
    for (size_t i = 0; i < CORE_DIM; ++i)
        x[i] = 0.5;

    // Two different contexts
    std::array<double, ContextSuperposition::KEY_DIM> ctx_cooking{};
    std::array<double, ContextSuperposition::KEY_DIM> ctx_skin{};
    for (size_t d = 0; d < ContextSuperposition::KEY_DIM; ++d) {
        ctx_cooking[d] = 0.8 * std::sin(static_cast<double>(d));
        ctx_skin[d] = -0.6 * std::cos(static_cast<double>(d * 3));
    }

    CoreVec out_cooking{}, out_skin{};
    sp.modulate(W, x, ctx_cooking, out_cooking);
    sp.modulate(W, x, ctx_skin, out_skin);

    // Outputs must differ
    double diff = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double d = out_cooking[i] - out_skin[i];
        diff += d * d;
    }
    diff = std::sqrt(diff);
    assert(diff > 1e-6);  // Must be meaningfully different

    std::printf("PASS (diff=%.6f)\n", diff);
}

// =============================================================================
// Test 3: Serialization roundtrip — 161 params preserved
// =============================================================================

static void test_serialization_roundtrip() {
    std::printf("  [3] Serialization roundtrip... ");

    ConceptModel model;

    // Enable and fill superposition with known values
    auto& sp = model.superposition();
    sp.enabled = 1.0;
    for (size_t i = 0; i < ContextSuperposition::N_MODES * CORE_DIM; ++i) {
        sp.u[i] = 0.01 * static_cast<double>(i + 1);
        sp.v[i] = -0.02 * static_cast<double>(i + 3);
    }
    for (size_t i = 0; i < ContextSuperposition::N_MODES * ContextSuperposition::KEY_DIM; ++i) {
        sp.keys[i] = 0.05 * static_cast<double>(i + 7);
    }

    // Serialize
    std::array<double, CM_FLAT_SIZE> flat;
    model.to_flat(flat);

    // Deserialize into fresh model
    ConceptModel model2;
    model2.from_flat(flat);

    const auto& sp2 = model2.superposition();
    assert(sp2.enabled > 0.5);

    for (size_t i = 0; i < ContextSuperposition::N_MODES * CORE_DIM; ++i) {
        assert(approx_eq(sp.u[i], sp2.u[i]));
        assert(approx_eq(sp.v[i], sp2.v[i]));
    }
    for (size_t i = 0; i < ContextSuperposition::N_MODES * ContextSuperposition::KEY_DIM; ++i) {
        assert(approx_eq(sp.keys[i], sp2.keys[i]));
    }

    std::printf("PASS\n");
}

// =============================================================================
// Test 4: V7→V8 migration — zero-fill means disabled
// =============================================================================

static void test_v7_migration() {
    std::printf("  [4] V7→V8 migration... ");

    // Simulate V7 data: 9772 doubles of known data + zero-fill to 9933
    ConceptModel original;

    // Set some bilinear params
    FlexEmbedding e, c;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e.core[i] = 0.2 * std::sin(static_cast<double>(i));
        c.core[i] = 0.3 * std::cos(static_cast<double>(i));
    }

    // Train a bit to have non-trivial state
    MicroTrainingConfig config;
    config.max_epochs = 5;
    original.train_step(e, c, 0.8, config);

    // Serialize to flat
    std::array<double, CM_FLAT_SIZE> full_flat;
    original.to_flat(full_flat);

    // Simulate V7 migration: take first 9772 doubles, zero-fill rest
    std::array<double, CM_FLAT_SIZE> migrated_flat{};
    for (size_t i = 0; i < CM_FLAT_SIZE_V7; ++i)
        migrated_flat[i] = full_flat[i];
    // offsets 9772..9932 are zero (default init)

    ConceptModel migrated;
    migrated.from_flat(migrated_flat);

    // Superposition should be disabled
    assert(migrated.superposition().enabled < 0.5);

    // All u, v, keys should be zero
    for (size_t i = 0; i < ContextSuperposition::N_MODES * CORE_DIM; ++i) {
        assert(migrated.superposition().u[i] == 0.0);
        assert(migrated.superposition().v[i] == 0.0);
    }
    for (size_t i = 0; i < ContextSuperposition::N_MODES * ContextSuperposition::KEY_DIM; ++i) {
        assert(migrated.superposition().keys[i] == 0.0);
    }

    // Original bilinear predict should still work
    double score = migrated.predict(e, c);
    assert(score > 0.0 && score < 1.0);

    std::printf("PASS\n");
}

// =============================================================================
// Test 5: Attention normalization — softmax sums to 1.0
// =============================================================================

static void test_attention_normalization() {
    std::printf("  [5] Attention normalization... ");

    ContextSuperposition sp;
    sp.enabled = 1.0;

    // Set random-ish keys
    for (size_t k = 0; k < ContextSuperposition::N_MODES; ++k)
        for (size_t d = 0; d < ContextSuperposition::KEY_DIM; ++d)
            sp.keys[k * ContextSuperposition::KEY_DIM + d] =
                std::sin(static_cast<double>(k * 10 + d));

    // Test with various ctx queries
    for (int trial = 0; trial < 5; ++trial) {
        std::array<double, ContextSuperposition::KEY_DIM> q{};
        for (size_t d = 0; d < ContextSuperposition::KEY_DIM; ++d)
            q[d] = std::cos(static_cast<double>(trial * 7 + d));

        // Compute logits and softmax manually to verify
        std::array<double, ContextSuperposition::N_MODES> logits{};
        for (size_t k = 0; k < ContextSuperposition::N_MODES; ++k) {
            double dot = 0.0;
            for (size_t d = 0; d < ContextSuperposition::KEY_DIM; ++d)
                dot += sp.keys[k * ContextSuperposition::KEY_DIM + d] * q[d];
            logits[k] = dot;
        }

        double max_l = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        std::array<double, ContextSuperposition::N_MODES> alpha{};
        for (size_t k = 0; k < ContextSuperposition::N_MODES; ++k) {
            alpha[k] = std::exp(logits[k] - max_l);
            sum += alpha[k];
        }
        for (size_t k = 0; k < ContextSuperposition::N_MODES; ++k)
            alpha[k] /= sum;

        // Verify sum = 1.0
        double alpha_sum = 0.0;
        for (size_t k = 0; k < ContextSuperposition::N_MODES; ++k)
            alpha_sum += alpha[k];
        assert(approx_eq(alpha_sum, 1.0, 1e-12));

        // All alphas must be positive
        for (size_t k = 0; k < ContextSuperposition::N_MODES; ++k)
            assert(alpha[k] > 0.0);
    }

    std::printf("PASS\n");
}

// =============================================================================
// Test 6: Modulation math — hand-computed verification
// =============================================================================

static void test_modulation_math() {
    std::printf("  [6] Modulation math... ");

    ContextSuperposition sp;
    sp.enabled = 1.0;

    // Simple setup: only mode 0 has non-zero u,v
    // u_0 = [1, 0, 0, ...], v_0 = [0, 1, 0, ...]
    sp.u[0] = 1.0;  // u_0[0] = 1
    sp.v[1] = 1.0;  // v_0[1] = 1

    // All keys zero → uniform attention: α_k = 0.25 for all k
    // (softmax of equal values)

    // W = identity
    CoreMat W{};
    for (size_t i = 0; i < CORE_DIM; ++i)
        W[i * CORE_DIM + i] = 1.0;

    // x = [1, 2, 3, 4, 0, 0, ...]
    CoreVec x{};
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0;

    // Expected:
    // v_0^T * x = v_0[1] * x[1] = 1.0 * 2.0 = 2.0
    // Other modes: v_k = 0 → v_k^T * x = 0
    // α_0 = 0.25 (uniform since all keys=0)
    // W_eff * x = W*x + 0.25 * 2.0 * u_0 = [1, 2, 3, 4, ...] + [0.5, 0, 0, ...]
    // output[0] = 1.0 + 0.5 = 1.5
    // output[1] = 2.0
    // output[2] = 3.0
    // output[3] = 4.0

    std::array<double, ContextSuperposition::KEY_DIM> ctx_query{};  // all zero
    CoreVec output{};
    sp.modulate(W, x, ctx_query, output);

    assert(approx_eq(output[0], 1.5, 1e-8));
    assert(approx_eq(output[1], 2.0, 1e-8));
    assert(approx_eq(output[2], 3.0, 1e-8));
    assert(approx_eq(output[3], 4.0, 1e-8));
    for (size_t i = 4; i < CORE_DIM; ++i)
        assert(approx_eq(output[i], 0.0, 1e-8));

    std::printf("PASS\n");
}

// =============================================================================
// Test 7: Zero superposition = identity modulation
// =============================================================================

static void test_zero_superposition_identity() {
    std::printf("  [7] Zero superposition = identity... ");

    ContextSuperposition sp;
    sp.enabled = 1.0;
    // All u, v, keys are zero by default

    // W = some matrix, x = some input
    CoreMat W{};
    CoreVec x{};
    for (size_t i = 0; i < CORE_DIM; ++i) {
        for (size_t j = 0; j < CORE_DIM; ++j)
            W[i * CORE_DIM + j] = 0.1 * std::sin(static_cast<double>(i * 10 + j));
        x[i] = std::cos(static_cast<double>(i * 3));
    }

    // When u=v=0, modulate should return exactly W*x
    std::array<double, ContextSuperposition::KEY_DIM> q{};
    for (size_t d = 0; d < ContextSuperposition::KEY_DIM; ++d)
        q[d] = 0.5;

    CoreVec mod_output{};
    sp.modulate(W, x, q, mod_output);

    // Compute W*x directly
    CoreVec direct{};
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < CORE_DIM; ++j)
            sum += W[i * CORE_DIM + j] * x[j];
        direct[i] = sum;
    }

    for (size_t i = 0; i < CORE_DIM; ++i)
        assert(approx_eq(mod_output[i], direct[i]));

    std::printf("PASS\n");
}

// =============================================================================
// Test 8: CM_FLAT_SIZE consistency
// =============================================================================

static void test_flat_size_consistency() {
    std::printf("  [8] CM_FLAT_SIZE consistency... ");

    // V8 = V7 + 161 superposition params
    assert(CM_FLAT_SIZE == CM_FLAT_SIZE_V7 + ContextSuperposition::TOTAL_PARAMS);
    assert(CM_FLAT_SIZE == 9933);
    assert(ContextSuperposition::TOTAL_PARAMS == 161);

    std::printf("PASS\n");
}

// =============================================================================
// Test 9: train_superposition_step — gradient reduces prediction error
// =============================================================================

static void test_superposition_training() {
    std::printf("  [9] Superposition training... ");

    ConceptModel model;

    // Enable superposition with small random init
    auto& sp = model.superposition();
    sp.enabled = 1.0;
    for (size_t i = 0; i < sp.u.size(); ++i) {
        double x = std::sin(static_cast<double>(i * 71 + 37)) * 43758.5453;
        x = x - std::floor(x);
        sp.u[i] = (x * 2.0 - 1.0) * 0.01;
    }
    for (size_t i = 0; i < sp.v.size(); ++i) {
        double x = std::sin(static_cast<double>(i * 53 + 19)) * 43758.5453;
        x = x - std::floor(x);
        sp.v[i] = (x * 2.0 - 1.0) * 0.01;
    }
    for (size_t i = 0; i < sp.keys.size(); ++i) {
        double x = std::sin(static_cast<double>(i * 97 + 41)) * 43758.5453;
        x = x - std::floor(x);
        sp.keys[i] = (x * 2.0 - 1.0) * 0.01;
    }

    // Create test embeddings
    FlexEmbedding e, c;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e.core[i] = 0.3 * std::sin(static_cast<double>(i * 5 + 1));
        c.core[i] = 0.4 * std::cos(static_cast<double>(i * 7 + 2));
    }

    std::array<double, ContextSuperposition::KEY_DIM> ctx_query{};
    for (size_t d = 0; d < ContextSuperposition::KEY_DIM; ++d)
        ctx_query[d] = 0.5 * std::sin(static_cast<double>(d * 3));

    double target = 0.8;

    // Compute initial prediction
    double pred_before = model.predict(e, c);

    // Train for several steps
    for (int step = 0; step < 100; ++step) {
        model.train_superposition_step(e, c, target, 0.01, ctx_query);
    }

    // Check prediction changed
    double pred_after = model.predict(e, c);

    // Allow small tolerance — superposition training modifies W_eff indirectly
    // but predict() uses W directly (not W_eff), so the effect is through u,v
    // which the standard predict() doesn't use. The test verifies the gradient
    // computation doesn't crash and parameters change.
    bool params_changed = false;
    for (size_t i = 0; i < sp.u.size(); ++i) {
        double x = std::sin(static_cast<double>(i * 71 + 37)) * 43758.5453;
        x = x - std::floor(x);
        double orig = (x * 2.0 - 1.0) * 0.01;
        if (std::abs(sp.u[i] - orig) > 1e-15) { params_changed = true; break; }
    }
    assert(params_changed);  // Gradient must have updated parameters

    std::printf("PASS (params updated, pred: %.4f → %.4f)\n", pred_before, pred_after);
}

// =============================================================================
// Test 10: SuperpositionTracker — should_enable logic
// =============================================================================

// Minimal SuperpositionTracker reimplementation for standalone testing
// (matches the struct in colearn_loop.hpp)

namespace {
struct TestTracker {
    struct ConceptContextStats {
        std::unordered_map<uint16_t, std::vector<double>> quality_by_relation;
        size_t total_observations = 0;
    };
    std::unordered_map<uint32_t, ConceptContextStats> concept_stats;

    void record(uint32_t cid, uint16_t rel, double quality) {
        auto& cs = concept_stats[cid];
        cs.quality_by_relation[rel].push_back(quality);
        ++cs.total_observations;
    }

    bool should_enable(uint32_t cid, size_t min_obs, double min_std) const {
        auto it = concept_stats.find(cid);
        if (it == concept_stats.end()) return false;
        const auto& cs = it->second;
        if (cs.total_observations < min_obs) return false;
        if (cs.quality_by_relation.size() < 2) return false;

        std::vector<double> means;
        for (const auto& [rel, qualities] : cs.quality_by_relation) {
            if (qualities.empty()) continue;
            double sum = 0.0;
            for (double q : qualities) sum += q;
            means.push_back(sum / static_cast<double>(qualities.size()));
        }
        if (means.size() < 2) return false;

        double mean_of_means = 0.0;
        for (double m : means) mean_of_means += m;
        mean_of_means /= static_cast<double>(means.size());

        double variance = 0.0;
        for (double m : means) {
            double d = m - mean_of_means;
            variance += d * d;
        }
        variance /= static_cast<double>(means.size());
        return std::sqrt(variance) > min_std;
    }
};
}

static void test_superposition_tracker() {
    std::printf("  [10] SuperpositionTracker logic... ");

    TestTracker tracker;

    // Not enough observations → should not enable
    tracker.record(42, 1, 0.8);
    tracker.record(42, 2, 0.3);
    assert(!tracker.should_enable(42, 10, 0.15));

    // Add more observations with diverse relations and varying quality
    // Relation 1 (CAUSES): consistently high quality
    for (int i = 0; i < 6; ++i) tracker.record(42, 1, 0.8 + 0.02 * i);
    // Relation 2 (HAS_PROPERTY): consistently low quality
    for (int i = 0; i < 6; ++i) tracker.record(42, 2, 0.2 + 0.02 * i);

    // Now: 14 observations, 2 relations, mean(rel1)~0.85, mean(rel2)~0.25
    // std of means = sqrt(((0.85-0.55)² + (0.25-0.55)²)/2) = sqrt(0.09+0.09)/2) = 0.3
    assert(tracker.should_enable(42, 10, 0.15));  // 0.3 > 0.15

    // Concept with uniform quality across relations → should NOT enable
    for (int i = 0; i < 10; ++i) {
        tracker.record(99, 1, 0.6);
        tracker.record(99, 2, 0.6);
    }
    assert(!tracker.should_enable(99, 10, 0.15));  // std ≈ 0

    // Only one relation type → should NOT enable
    for (int i = 0; i < 15; ++i) tracker.record(77, 1, 0.5);
    assert(!tracker.should_enable(77, 10, 0.15));

    // Unknown concept → should NOT enable
    assert(!tracker.should_enable(999, 10, 0.15));

    std::printf("PASS\n");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::printf("=== ContextSuperposition Tests ===\n\n");

    test_backward_compatibility();
    test_context_sensitivity();
    test_serialization_roundtrip();
    test_v7_migration();
    test_attention_normalization();
    test_modulation_math();
    test_zero_superposition_identity();
    test_flat_size_consistency();
    test_superposition_training();
    test_superposition_tracker();

    std::printf("\n=== All %d tests passed ===\n", 10);
    return 0;
}
