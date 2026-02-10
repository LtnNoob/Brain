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
        {1, 2},
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

    // H2: With extracted data, trust can be high
    auto assessment = bridge.assess(hyp, train_result, train_config,
                                     DataQuality::EXTRACTED, 100);

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
    // H2: With synthetic data default, trust is further reduced
    ASSERT(assessment.metadata.trust >= 0.0);
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

    ASSERT(result.pattern == RelationshipPattern::LINEAR);
    ASSERT(!result.explanation.empty());
}

// =============================================================================
// TEST 7: DomainManager — Domain detection from relations
// =============================================================================

void test_domain_detection() {
    DomainManager manager;
    LongTermMemory ltm;

    auto c1 = ltm.store_concept("Force", "Physical force",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c2 = ltm.store_concept("Acceleration", "Rate of velocity change",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c3 = ltm.store_concept("Mass", "Amount of matter",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    ltm.add_relation(c1, c2, RelationType::CAUSES, 0.9);
    ltm.add_relation(c1, c3, RelationType::HAS_PROPERTY, 0.8);

    auto domain = manager.detect_domain(c1, ltm);
    ASSERT(domain == DomainType::PHYSICAL);

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

    auto p1 = ltm.store_concept("Gravity", "Gravitational force",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto p2 = ltm.store_concept("Weight", "Force due to gravity",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    ltm.add_relation(p1, p2, RelationType::CAUSES, 0.9);
    ltm.add_relation(p1, p2, RelationType::HAS_PROPERTY, 0.8);

    auto s1 = ltm.store_concept("Opinion", "Personal belief",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    auto s2 = ltm.store_concept("Debate", "Discussion of opposing views",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.5));
    ltm.add_relation(s1, s2, RelationType::SUPPORTS, 0.7);
    ltm.add_relation(s2, s1, RelationType::CONTRADICTS, 0.6);

    ltm.add_relation(p1, s1, RelationType::SIMILAR_TO, 0.3);

    auto insights = manager.find_cross_domain_insights(
        {p1, p2, s1, s2}, ltm
    );

    // Fixed: was ASSERT(insights.size() >= 0) which is always true for size_t
    // Now we just verify no crash and reasonable output
    (void)insights; // no-crash test
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
    loop_config.mse_threshold = 0.5;
    loop_config.improvement_threshold = 0.0001;

    RefinementLoop loop(std::move(validator), loop_config);

    auto initial = make_hypothesis(9, "X increases linearly with Y", {"linear"});

    auto refiner = [](const HypothesisProposal& prev, const std::string& /*feedback*/) {
        return prev;
    };

    auto result = loop.run(initial, refiner);

    ASSERT(result.iterations_performed > 0);
    ASSERT(result.iterations_performed <= loop_config.max_iterations);
    ASSERT(result.provenance_chain.size() == result.iterations_performed);
    for (size_t i = 0; i < result.provenance_chain.size(); ++i) {
        ASSERT(result.provenance_chain[i].iteration_number == i);
    }
}

// =============================================================================
// TEST 10: RefinementLoop — Max iterations termination
// =============================================================================

void test_refinement_max_iterations() {
    KanValidator::Config val_config;
    val_config.max_epochs = 100;
    KanValidator validator(val_config);

    RefinementLoop::Config loop_config;
    loop_config.max_iterations = 3;
    loop_config.mse_threshold = 0.0001;
    loop_config.improvement_threshold = 0.0;

    RefinementLoop loop(std::move(validator), loop_config);

    auto initial = make_hypothesis(10, "X oscillates periodically with Y", {"periodic"});

    auto refiner = [](const HypothesisProposal& prev, const std::string& /*feedback*/) {
        return prev;
    };

    auto result = loop.run(initial, refiner);

    // Should either hit max iterations or stall
    ASSERT(result.iterations_performed <= loop_config.max_iterations);
    ASSERT(result.provenance_chain.size() == result.iterations_performed);
}

// =============================================================================
// TEST 11: Pattern detection (original)
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
// TEST 12: EpistemicBridge — HYPOTHESIS range
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

    // With extracted data, original range applies
    auto assessment = bridge.assess(hyp, train_result, train_config,
                                     DataQuality::EXTRACTED, 100);

    ASSERT(assessment.metadata.type == EpistemicType::HYPOTHESIS);
    ASSERT(assessment.metadata.trust >= 0.4);
    ASSERT(assessment.metadata.trust <= 0.7);
}

// =============================================================================
// TEST 13: C1 — Negation-aware pattern detection
// =============================================================================

void test_negation_detection() {
    HypothesisTranslator translator;

    // "not exponential" should NOT be detected as EXPONENTIAL
    auto result = translator.detect_pattern_detailed("the growth is not exponential");
    ASSERT(result.pattern != RelationshipPattern::EXPONENTIAL);

    // "not periodic" should NOT be PERIODIC
    auto result2 = translator.detect_pattern_detailed("the signal is not periodic at all");
    ASSERT(result2.pattern != RelationshipPattern::PERIODIC);

    // Positive case still works
    auto result3 = translator.detect_pattern_detailed("the growth is exponential");
    ASSERT(result3.pattern == RelationshipPattern::EXPONENTIAL);
    ASSERT(result3.confidence > 0.5);
}

// =============================================================================
// TEST 14: C1 — Confidence scoring
// =============================================================================

void test_confidence_scoring() {
    HypothesisTranslator translator;

    // Strong signal → high confidence
    auto strong = translator.detect_pattern_detailed("exponential growth with geometric progression");
    ASSERT(strong.confidence > 0.5);

    // Weak/hedged signal → lower confidence
    auto weak = translator.detect_pattern_detailed("it sometimes increases a bit more");
    // "sometimes" → quantifier modifier 0.5, "increases" → weak LINEAR 0.3
    ASSERT(weak.confidence < 0.5);

    // Below threshold → NOT_QUANTIFIABLE
    auto vague = translator.detect_pattern_detailed("things might be slightly related");
    ASSERT(vague.pattern == RelationshipPattern::NOT_QUANTIFIABLE);
}

// =============================================================================
// TEST 15: C1 — Conditional pattern detection
// =============================================================================

void test_conditional_detection() {
    HypothesisTranslator translator;

    auto result = translator.detect_pattern_detailed("X increases when Y exceeds the threshold");
    // Should detect CONDITIONAL or THRESHOLD
    ASSERT(result.pattern == RelationshipPattern::CONDITIONAL ||
           result.pattern == RelationshipPattern::THRESHOLD);
    ASSERT(result.confidence > 0.3);
}

// =============================================================================
// TEST 16: C1 — Quantifier modifier
// =============================================================================

void test_quantifier_modifier() {
    HypothesisTranslator translator;

    auto always_result = translator.detect_pattern_detailed("X always increases linearly with Y");
    auto sometimes_result = translator.detect_pattern_detailed("X sometimes increases linearly with Y");

    // "always" should give higher confidence than "sometimes"
    ASSERT(always_result.confidence > sometimes_result.confidence);
}

// =============================================================================
// TEST 17: H1 — Numeric hint extraction
// =============================================================================

void test_numeric_hint_extraction() {
    HypothesisTranslator translator;

    auto hints = translator.extract_numeric_hints("Temperature increases at a rate of 2.5 between 0 and 100");
    ASSERT(hints.has_hints());
    ASSERT(hints.numbers.size() >= 3);  // 2.5, 0, 100

    // Range extraction
    ASSERT(hints.range_min.has_value());
    ASSERT(hints.range_max.has_value());
    ASSERT(std::abs(hints.range_min.value() - 0.0) < 0.01);
    ASSERT(std::abs(hints.range_max.value() - 100.0) < 0.01);
}

// =============================================================================
// TEST 18: H1 — Hypothesis-specific data quality tracking
// =============================================================================

void test_data_quality_tracking() {
    HypothesisTranslator translator;

    // With numeric hints → SYNTHETIC_SPECIFIC
    auto proposal1 = make_hypothesis(18, "X increases linearly with slope of 3.0 between 0 and 10",
                                      {"linear"});
    auto result1 = translator.translate(proposal1);
    ASSERT(result1.translatable);
    ASSERT(result1.problem->data_quality == DataQuality::SYNTHETIC_SPECIFIC);

    // Without numeric hints → SYNTHETIC_CANONICAL
    auto proposal2 = make_hypothesis(19, "X increases linearly with Y", {"linear"});
    auto result2 = translator.translate(proposal2);
    ASSERT(result2.translatable);
    ASSERT(result2.problem->data_quality == DataQuality::SYNTHETIC_CANONICAL);
}

// =============================================================================
// TEST 19: H2 — Trust-inflation cap for synthetic data
// =============================================================================

void test_trust_inflation_cap() {
    EpistemicBridge bridge;

    auto kan = std::make_shared<KANModule>(1, 1, 10);
    FunctionHypothesis hyp(1, 1, kan, 100, 0.001);  // Excellent MSE

    KanTrainingResult train_result;
    train_result.iterations_run = 100;
    train_result.final_loss = 0.001;
    train_result.converged = true;
    train_result.duration = std::chrono::milliseconds(50);

    KanTrainingConfig train_config;
    train_config.max_iterations = 1000;

    // H2: With SYNTHETIC_CANONICAL data, trust must be capped at 0.6
    auto assessment = bridge.assess(hyp, train_result, train_config,
                                     DataQuality::SYNTHETIC_CANONICAL, 100);

    ASSERT(assessment.metadata.trust <= 0.6);
    ASSERT(assessment.data_quality == DataQuality::SYNTHETIC_CANONICAL);
}

// =============================================================================
// TEST 20: H2 — Trivial convergence penalty
// =============================================================================

void test_trivial_convergence_penalty() {
    EpistemicBridge bridge;

    auto kan = std::make_shared<KANModule>(1, 1, 10);
    FunctionHypothesis hyp(1, 1, kan, 5, 0.001);  // Very fast convergence (5 iters)

    KanTrainingResult train_result_fast;
    train_result_fast.iterations_run = 5;  // < 10 = trivial
    train_result_fast.final_loss = 0.001;
    train_result_fast.converged = true;
    train_result_fast.duration = std::chrono::milliseconds(5);

    KanTrainingResult train_result_normal;
    train_result_normal.iterations_run = 200;  // Normal convergence
    train_result_normal.final_loss = 0.001;
    train_result_normal.converged = true;
    train_result_normal.duration = std::chrono::milliseconds(100);

    KanTrainingConfig train_config;
    train_config.max_iterations = 1000;

    auto fast_assessment = bridge.assess(hyp, train_result_fast, train_config,
                                          DataQuality::EXTRACTED, 100);
    auto normal_assessment = bridge.assess(hyp, train_result_normal, train_config,
                                            DataQuality::EXTRACTED, 100);

    // Trivially convergent should have lower trust
    ASSERT(fast_assessment.metadata.trust < normal_assessment.metadata.trust);
}

// =============================================================================
// TEST 21: H2 — Minimum data points for high trust
// =============================================================================

void test_min_data_points_trust() {
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

    // Few data points → trust capped at 0.5
    auto assessment_few = bridge.assess(hyp, train_result, train_config,
                                         DataQuality::EXTRACTED, 20);
    ASSERT(assessment_few.metadata.trust <= 0.5);

    // Enough data points → trust can be higher
    auto assessment_many = bridge.assess(hyp, train_result, train_config,
                                          DataQuality::EXTRACTED, 100);
    ASSERT(assessment_many.metadata.trust > 0.5);
}

// =============================================================================
// TEST 22: Division by zero guard (n=1)
// =============================================================================

void test_division_by_zero_guard() {
    HypothesisTranslator translator;

    // n=1 should not crash
    auto data = translator.generate_training_data(RelationshipPattern::LINEAR, 1, 0.0, 1.0);
    // Should return empty (guarded)
    ASSERT(data.empty());

    // n=0 should not crash
    auto data0 = translator.generate_training_data(RelationshipPattern::LINEAR, 0, 0.0, 1.0);
    ASSERT(data0.empty());
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

    // New tests for fixes
    std::cout << "\n--- C1: NLP-lite Parser Tests ---\n";
    TEST(negation_detection);
    TEST(confidence_scoring);
    TEST(conditional_detection);
    TEST(quantifier_modifier);

    std::cout << "\n--- H1: Hypothesis-Specific Data Tests ---\n";
    TEST(numeric_hint_extraction);
    TEST(data_quality_tracking);

    std::cout << "\n--- H2: Trust-Inflation Cap Tests ---\n";
    TEST(trust_inflation_cap);
    TEST(trivial_convergence_penalty);
    TEST(min_data_points_trust);

    std::cout << "\n--- Edge Case Tests ---\n";
    TEST(division_by_zero_guard);

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
