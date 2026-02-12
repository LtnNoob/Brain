// Unit Test: Termination Logic + Conflict Resolution
//
// Build:
//   make test_termination_conflict

#include "termination.hpp"
#include "conflict_resolution.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace brain19;

// =============================================================================
// Test 1: Termination — depth limit
// =============================================================================
void test_termination_depth() {
    std::cout << "TEST: Termination depth limit... ";

    GoalState goal;
    FocusCursorConfig config;
    config.max_depth = 5;

    // At depth 4 — not terminated
    CursorView view{0, 4, {}, 0.0, ExplorationMode::GREEDY, 5};
    assert(!check_termination(goal, view, config));

    // At depth 5 — terminated
    view.depth = 5;
    assert(check_termination(goal, view, config));

    // At depth 10 — definitely terminated
    view.depth = 10;
    assert(check_termination(goal, view, config));

    std::cout << "PASS\n";
}

// =============================================================================
// Test 2: Termination — energy budget
// =============================================================================
void test_termination_energy() {
    std::cout << "TEST: Termination energy budget... ";

    GoalState goal;
    FocusCursorConfig config;
    config.max_depth = 100;  // High so depth doesn't trigger
    config.energy_budget = 5.0;

    // Under budget
    CursorView view{0, 2, {}, 4.9, ExplorationMode::GREEDY, 3};
    assert(!check_termination(goal, view, config));

    // At budget
    view.accumulated_energy = 5.0;
    assert(check_termination(goal, view, config));

    // Over budget
    view.accumulated_energy = 7.0;
    assert(check_termination(goal, view, config));

    std::cout << "PASS\n";
}

// =============================================================================
// Test 3: Termination — goal completion (GOAL_DIRECTED only)
// =============================================================================
void test_termination_goal() {
    std::cout << "TEST: Termination goal completion... ";

    FocusCursorConfig config;
    config.max_depth = 100;
    config.energy_budget = 100.0;

    // Create a definition goal that's already complete
    Vec10 emb{};
    GoalState goal = GoalState::definition_goal(42, emb, "find X");
    goal.update_progress({1, 2, 42}, 3);  // Target found
    assert(goal.is_complete());

    // In GREEDY mode — goal completion doesn't trigger termination
    CursorView view{0, 2, {}, 1.0, ExplorationMode::GREEDY, 3};
    assert(!check_termination(goal, view, config));

    // In GOAL_DIRECTED mode — goal completion triggers termination
    view.mode = ExplorationMode::GOAL_DIRECTED;
    assert(check_termination(goal, view, config));

    // Incomplete goal in GOAL_DIRECTED mode — no termination
    GoalState incomplete = GoalState::definition_goal(99, emb, "find Y");
    assert(!incomplete.is_complete());
    assert(!check_termination(incomplete, view, config));

    std::cout << "PASS\n";
}

// =============================================================================
// Test 4: Termination — no trigger when all conditions are fine
// =============================================================================
void test_termination_no_trigger() {
    std::cout << "TEST: Termination no trigger... ";

    GoalState goal;
    FocusCursorConfig config;
    config.max_depth = 12;
    config.energy_budget = 10.0;

    CursorView view{0, 3, {}, 2.0, ExplorationMode::EXPLORATORY, 4};
    assert(!check_termination(goal, view, config));

    std::cout << "PASS\n";
}

// =============================================================================
// Test 5: Conflict Resolution — effective_priority
// =============================================================================
void test_effective_priority() {
    std::cout << "TEST: Conflict effective_priority... ";

    EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);

    ConceptInfo a(1, "A", "desc", meta);
    a.structural_confidence = 0.8;
    a.semantic_confidence = 0.6;
    a.activation_score = 1.0;

    // Default weights: alpha=0.4, beta=0.4, gamma=0.2
    // expected = 0.4*0.8 + 0.4*0.6 + 0.2*1.0 = 0.32 + 0.24 + 0.20 = 0.76
    double p = effective_priority(a);
    assert(std::abs(p - 0.76) < 0.001);

    // Custom weights
    ConflictWeights w{0.5, 0.3, 0.2};
    // expected = 0.5*0.8 + 0.3*0.6 + 0.2*1.0 = 0.40 + 0.18 + 0.20 = 0.78
    p = effective_priority(a, w);
    assert(std::abs(p - 0.78) < 0.001);

    std::cout << "PASS\n";
}

// =============================================================================
// Test 6: Conflict Resolution — resolves_in_favor
// =============================================================================
void test_resolves_in_favor() {
    std::cout << "TEST: Conflict resolves_in_favor... ";

    EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);

    ConceptInfo strong(1, "Strong", "high confidence", meta);
    strong.structural_confidence = 0.9;
    strong.semantic_confidence = 0.8;
    strong.activation_score = 0.7;

    ConceptInfo weak(2, "Weak", "low confidence", meta);
    weak.structural_confidence = 0.2;
    weak.semantic_confidence = 0.1;
    weak.activation_score = 0.3;

    // Strong should win
    assert(resolves_in_favor(strong, weak));
    assert(!resolves_in_favor(weak, strong));

    // Equal concepts — neither wins
    ConceptInfo equal(3, "Equal", "same as strong", meta);
    equal.structural_confidence = 0.9;
    equal.semantic_confidence = 0.8;
    equal.activation_score = 0.7;
    assert(!resolves_in_favor(strong, equal));  // Not strictly greater
    assert(!resolves_in_favor(equal, strong));

    std::cout << "PASS\n";
}

// =============================================================================
// Test 7: Conflict Resolution — zero scores
// =============================================================================
void test_conflict_zero_scores() {
    std::cout << "TEST: Conflict zero scores... ";

    EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);

    ConceptInfo zero(1, "Zero", "no confidence", meta);
    // All new fields default to 0.0

    double p = effective_priority(zero);
    assert(std::abs(p) < 0.001);

    std::cout << "PASS\n";
}

// =============================================================================
// Test 8: Conflict Resolution — custom weights sum to 1.0 check
// =============================================================================
void test_conflict_custom_weights() {
    std::cout << "TEST: Conflict custom weights... ";

    EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);

    ConceptInfo c(1, "C", "test", meta);
    c.structural_confidence = 1.0;
    c.semantic_confidence = 1.0;
    c.activation_score = 1.0;

    // With all-1.0 scores, priority equals sum of weights
    ConflictWeights w{0.5, 0.3, 0.2};
    double p = effective_priority(c, w);
    assert(std::abs(p - 1.0) < 0.001);

    // Extreme: all weight on activation
    ConflictWeights w2{0.0, 0.0, 1.0};
    c.activation_score = 0.42;
    p = effective_priority(c, w2);
    assert(std::abs(p - 0.42) < 0.001);

    std::cout << "PASS\n";
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n=== Termination + Conflict Resolution Tests ===\n\n";

    test_termination_depth();
    test_termination_energy();
    test_termination_goal();
    test_termination_no_trigger();
    test_effective_priority();
    test_resolves_in_favor();
    test_conflict_zero_scores();
    test_conflict_custom_weights();

    std::cout << "\n=== ALL TESTS PASSED ===\n\n";
    return 0;
}
