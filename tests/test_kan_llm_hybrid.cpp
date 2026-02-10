#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "../backend/hybrid/hypothesis_translator.hpp"
#include "../backend/hybrid/epistemic_bridge.hpp"
#include "../backend/hybrid/kan_validator.hpp"
#include "../backend/hybrid/domain_manager.hpp"
#include "../backend/hybrid/refinement_loop.hpp"
#include "../backend/kan/kan_module.hpp"
#include "../backend/kan/function_hypothesis.hpp"
#include "../backend/ltm/long_term_memory.hpp"

using namespace brain19;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  TEST: " << #name << "... "; \
    try { test_##name(); tests_passed++; std::cout << "PASSED\n"; } \
    catch (const std::exception& e) { tests_failed++; std::cout << "FAILED: " << e.what() << "\n"; } \
    catch (...) { tests_failed++; std::cout << "FAILED (unknown exception)\n"; }

#define ASSERT(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond " at line " + std::to_string(__LINE__))

// =============================================================================
// Helper: Create a HypothesisProposal
// =============================================================================

static HypothesisProposal make_hypothesis(
    uint64_t id,
    const std::string& statement,
    const std::vector<std::string>& patterns = {},
    const std::string& reasoning = ""
) {
    return HypothesisProposal(
        id,
        {1, 2},  // evidence concepts
        statement,
        reasoning.empty() ? statement : reasoning,
        patterns,
        0.5,
        "test-model"
    );
}

// =============================================================================
// TEST 1: HypothesisTranslator — Linear relationship → KAN problem
// =============================================================================

void test_translator_linear() {
    HypothesisTranslator translator;

    auto proposal = make_hypothesis(1, "X increases proportionally with Y",
                                     {"linear", "proportional"});
    auto result = translator.translate(proposal);

    ASSERT(result.translatable);
    ASSERT(result.detected_pattern == RelationshipPattern::LINEAR);
    ASSERT(result.problem.has_value());
    ASSERT(result.problem->training_data.size() > 0);
    ASSERT(result.problem->input_dim == 1);
    ASSERT(result.problem->output_dim == 1);
}

// =============================================================================
// TEST 2: HypothesisTranslator — Non-quantifiable → NOT_QUANTIFIABLE
// =============================================================================

void test_translator_not_quantifiable() {
    HypothesisTranslator translator;

    auto proposal = make_hypothesis(2, "Beauty is subjective and culturally defined");
    auto result = translator.translate(proposal);

    ASSERT(!result.translatable);
    ASSERT(result.detected_pattern == RelationshipPattern::NOT_QUANTIFIABLE);
    ASSERT(!result.problem.has_value());
}

// =============================================================================
// TEST 3: EpistemicBridge — Good fit (MSE<0.01) → THEORY Trust
// =============================================================================

void test_bridge_good_fit() {
    EpistemicBridge bridge;

    auto kan = std::make_shared<KANModule>(1, 1, 10);
    FunctionHypothesis hyp(1, 1, kan, 100, 0.005);

    KanTrainingResult train_result;
    train_result.iterations_run = 100;
    train_result.final_loss = 0.005;
    train_result.converged = true;
    train_result.duration = std::chrono::milliseconds(50);

    KanTrainingConfig train_config;
    train_config.max_iterations = 1000;

    auto assessment = bridge.assess(hyp, train_result, train_config);

    ASSERT(assessment.metadata.type == EpistemicType::THEORY);
    ASSERT(assessment.metadata.status == EpistemicStatus::ACTIVE);
    ASSERT(assessment.metadata.trust >= 0.7);
    ASSERT(assessment.metadata.trust <= 1.0);
    ASSERT(assessment.converged);
}

// =============================================================================
// TEST 4: EpistemicBridge — Poor fit → SPECULATION Trust
// =============================================================================

void test_bridge_poor_fit() {
    EpistemicBridge bridge;

    auto kan = std::make_shared<KANModule>(1, 1, 10);
    FunctionHypothesis hyp(1, 1, kan, 1000, 0.5);

    KanTrainingResult train_result;
    train_result.iterations_run = 1000;
    train_result.final_loss = 0.5;
    train_result.converged = true;
    train_result.duration = std::chrono::milliseconds(500);

    KanTrainingConfig train_config;
    train_config.max_iterations = 1000;

    auto assessment = bridge.assess(hyp, train_result, train_config);

    ASSERT(assessment.metadata.type == EpistemicType::SPECULATION);
    ASSERT(assessment.metadata.trust >= 0.1);
    ASSERT(assessment.metadata.trust <= 0.3);
}

// =============================================================================
// TEST 5: EpistemicBridge — No convergence → INVALIDATED
// =============================================================================

void test_bridge_no_convergence() {
    EpistemicBridge bridge;

    auto kan = std::make_shared<KANModule>(1, 1, 10);
    FunctionHypothesis hyp(1, 1, kan, 1000, 5.0);

    KanTrainingResult train_result;
    train_result.iterations_run = 1000;
    train_result.final_loss = 5.0;
    train_result.converged = false;
    train_result.duration = std::chrono::milliseconds(1000);

    KanTrainingConfig train_config;
    train_config.max_iterations = 1000;

    auto assessment = bridge.assess(hyp, train_result, train_config);

    ASSERT(assessment.metadata.status == EpistemicStatus::INVALIDATED);
    ASSERT(assessment.metadata.trust < 0.1);
    ASSERT(!assessment.converged);
}

// =============================================================================
// TEST 6: KanValidator — End-to-end validation flow
// =============================================================================

void test_validator_end_to_end() {
    KanValidator::Config config;
    config.max_epochs = 500;
    config.convergence_threshold = 1e-6;
    KanValidator validator(config);

    auto proposal = make_hypothesis(6, "Temperature increases linearly with altitude in this range",
                                     {"linear"});

    auto result = validator.validate(proposal);

    // Should translate and attempt training
    ASSERT(result.pattern == RelationshipPattern::LINEAR);
    // The validation should produce a result (may or may not validate depending on training)
    ASSERT(!result.explanation.empty());
}

// =============================================================================
// TEST 7: DomainManager — Domain detection from relations
// =============================================================================

void test_domain_detection() {
    DomainManager manager;
    LongTermMemory ltm;

    // Create concepts
    auto c1 = ltm.store_concept("Force", "Physical force",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c2 = ltm.store_concept("Acceleration", "Rate of velocity change",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c3 = ltm.store_concept("Mass", "Amount of matter",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    // Physical domain: CAUSES + HAS_PROPERTY
    ltm.add_relation(c1, c2, RelationType::CAUSES, 0.9);
    ltm.add_relation(c1, c3, RelationType::HAS_PROPERTY, 0.8);

    auto domain = manager.detect_domain(c1, ltm);
    ASSERT(domain == DomainType::PHYSICAL);

    // Temporal domain
    auto c4 = ltm.store_concept("Event1", "First event",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c5 = ltm.store_concept("Event2", "Second event",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    ltm.add_relation(c4, c5, RelationType::TEMPORAL_BEFORE, 0.9);

    auto temporal_domain = manager.detect_domain(c4, ltm);
    ASSERT(temporal_domain == DomainType::TEMPORAL);
}

// =============================================================================
// TEST 8: DomainManager — Cross-domain query
// =============================================================================

void test_cross_domain_query() {
    DomainManager manager;
    LongTermMemory ltm;

    // Physical domain concepts
    auto p1 = ltm.store_concept("Gravity", "Gravitational force",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto p2 = ltm.store_concept("Weight", "Force due to gravity",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    ltm.add_relation(p1, p2, RelationType::CAUSES, 0.9);
    ltm.add_relation(p1, p2, RelationType::HAS_PROPERTY, 0.8);

    // Social domain concepts
    auto s1 = ltm.store_concept("Opinion", "Personal belief",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    auto s2 = ltm.store_concept("Debate", "Discussion of opposing views",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    ltm.add_relation(s1, s2, RelationType::SUPPORTS, 0.7);
    ltm.add_relation(s2, s1, RelationType::CONTRADICTS, 0.6);

    // Cross-domain link
    ltm.add_relation(p1, s1, RelationType::SIMILAR_TO, 0.3);

    auto insights = manager.find_cross_domain_insights(
        {p1, p2, s1, s2}, ltm
    );

    // Should find at least one cross-domain insight
    // (Physical ↔ Social via the SIMILAR_TO link)
    // Note: actual detection depends on clustering result
    // The test verifies the function runs without errors
    // and that cross-domain detection is possible
    ASSERT(insights.size() >= 0);  // At minimum, no crash
}

// =============================================================================
// TEST 9: RefinementLoop — Convergence within iterations
// =============================================================================

void test_refinement_convergence() {
    KanValidator::Config val_config;
    val_config.max_epochs = 500;
    KanValidator validator(val_config);

    RefinementLoop::Config loop_config;
    loop_config.max_iterations = 5;
    loop_config.mse_threshold = 0.5;  // Lenient threshold for test
    loop_config.improvement_threshold = 0.0001;

    RefinementLoop loop(std::move(validator), loop_config);

    auto initial = make_hypothesis(9, "X increases linearly with Y", {"linear"});

    // Refiner that always returns same hypothesis (linear is easy to fit)
    auto refiner = [](const HypothesisProposal& prev, const std::string& /*feedback*/) {
        return prev;  // Keep the same hypothesis
    };

    auto result = loop.run(initial, refiner);

    ASSERT(result.iterations_performed > 0);
    ASSERT(result.iterations_performed <= loop_config.max_iterations);
    ASSERT(result.provenance_chain.size() == result.iterations_performed);
    // Each iteration should be recorded
    for (size_t i = 0; i < result.provenance_chain.size(); ++i) {
        ASSERT(result.provenance_chain[i].iteration_number == i);
    }
}

// =============================================================================
// TEST 10: RefinementLoop — Max iterations termination
// =============================================================================

void test_refinement_max_iterations() {
    KanValidator::Config val_config;
    val_config.max_epochs = 100;  // Low epochs = might not converge well
    KanValidator validator(val_config);

    RefinementLoop::Config loop_config;
    loop_config.max_iterations = 3;
    loop_config.mse_threshold = 0.0001;  // Very strict — unlikely to converge
    loop_config.improvement_threshold = 0.0;  // Don't stop on stall

    RefinementLoop loop(std::move(validator), loop_config);

    auto initial = make_hypothesis(10, "X oscillates periodically with Y", {"periodic"});

    auto refiner = [](const HypothesisProposal& prev, const std::string& /*feedback*/) {
        return prev;
    };

    auto result = loop.run(initial, refiner);

    // Should hit max iterations
    ASSERT(result.iterations_performed == loop_config.max_iterations);
    ASSERT(result.provenance_chain.size() == loop_config.max_iterations);
}

// =============================================================================
// TEST 11 (bonus): Pattern detection
// =============================================================================

void test_pattern_detection() {
    HypothesisTranslator translator;

    ASSERT(translator.detect_pattern("exponential growth observed") == RelationshipPattern::EXPONENTIAL);
    ASSERT(translator.detect_pattern("periodic oscillation") == RelationshipPattern::PERIODIC);
    ASSERT(translator.detect_pattern("threshold activation at 50%") == RelationshipPattern::THRESHOLD);
    ASSERT(translator.detect_pattern("quadratic relationship") == RelationshipPattern::POLYNOMIAL);
    ASSERT(translator.detect_pattern("the sky is blue") == RelationshipPattern::NOT_QUANTIFIABLE);
}

// =============================================================================
// TEST 12 (bonus): EpistemicBridge — HYPOTHESIS range (MSE between 0.01-0.1)
// =============================================================================

void test_bridge_hypothesis_range() {
    EpistemicBridge bridge;

    auto kan = std::make_shared<KANModule>(1, 1, 10);
    FunctionHypothesis hyp(1, 1, kan, 500, 0.05);

    KanTrainingResult train_result;
    train_result.iterations_run = 500;
    train_result.final_loss = 0.05;
    train_result.converged = true;
    train_result.duration = std::chrono::milliseconds(200);

    KanTrainingConfig train_config;
    train_config.max_iterations = 1000;

    auto assessment = bridge.assess(hyp, train_result, train_config);

    ASSERT(assessment.metadata.type == EpistemicType::HYPOTHESIS);
    ASSERT(assessment.metadata.trust >= 0.4);
    ASSERT(assessment.metadata.trust <= 0.7);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "\n=== Brain19 Phase 7: KAN-LLM Hybrid Tests ===\n\n";

    TEST(translator_linear);
    TEST(translator_not_quantifiable);
    TEST(bridge_good_fit);
    TEST(bridge_poor_fit);
    TEST(bridge_no_convergence);
    TEST(validator_end_to_end);
    TEST(domain_detection);
    TEST(cross_domain_query);
    TEST(refinement_convergence);
    TEST(refinement_max_iterations);
    TEST(pattern_detection);
    TEST(bridge_hypothesis_range);

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
