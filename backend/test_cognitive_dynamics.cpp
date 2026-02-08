#include "cognitive/cognitive_dynamics.hpp"
#include "memory/brain_controller.hpp"
#include "ltm/long_term_memory.hpp"
#include "epistemic/epistemic_metadata.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace brain19;

int test_count = 0;
int passed_count = 0;

void print_test_header(const std::string& test_name) {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "TEST " << (++test_count) << ": " << test_name << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

void test_pass(const std::string& message) {
    std::cout << "✓ PASS: " << message << "\n";
    passed_count++;
}

void test_fail(const std::string& message) {
    std::cout << "✗ FAIL: " << message << "\n";
}

// =============================================================================
// TEST 1: Epistemic Invariants (CRITICAL)
// =============================================================================
void test_epistemic_invariants() {
    print_test_header("Epistemic Invariants Preservation");

    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;

    brain.initialize();

    // Create concepts with explicit epistemic metadata
    auto cat = ltm.store_concept("Cat", "Feline",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    auto dog = ltm.store_concept("Dog", "Canine",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.97));

    auto mystery = ltm.store_concept("Mystery", "Unknown",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.35));

    ltm.add_relation(cat, dog, RelationType::SIMILAR_TO, 0.8);

    // Get initial state
    auto cat_before = ltm.retrieve_concept(cat);
    auto dog_before = ltm.retrieve_concept(dog);
    auto mystery_before = ltm.retrieve_concept(mystery);

    // Spread activation
    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    cognitive.spread_activation(cat, 1.0, ctx, ltm, *stm);

    // Get state after spreading
    auto cat_after = ltm.retrieve_concept(cat);
    auto dog_after = ltm.retrieve_concept(dog);
    auto mystery_after = ltm.retrieve_concept(mystery);

    // CRITICAL TESTS
    assert(cat_after.has_value());
    assert(dog_after.has_value());
    assert(mystery_after.has_value());

    // Trust MUST be unchanged
    assert(std::abs(cat_after->epistemic.trust - 0.98) < 0.0001);
    assert(std::abs(dog_after->epistemic.trust - 0.97) < 0.0001);
    assert(std::abs(mystery_after->epistemic.trust - 0.35) < 0.0001);

    // Type MUST be unchanged
    assert(cat_after->epistemic.type == EpistemicType::FACT);
    assert(dog_after->epistemic.type == EpistemicType::FACT);
    assert(mystery_after->epistemic.type == EpistemicType::SPECULATION);

    // Status MUST be unchanged
    assert(cat_after->epistemic.status == EpistemicStatus::ACTIVE);

    test_pass("Trust values preserved");
    test_pass("EpistemicType preserved");
    test_pass("EpistemicStatus preserved");

    brain.destroy_context(ctx);
    brain.shutdown();
}

// =============================================================================
// TEST 2: Spreading Determinism
// =============================================================================
void test_spreading_determinism() {
    print_test_header("Spreading Activation Determinism");

    BrainController brain1, brain2;
    LongTermMemory ltm1, ltm2;
    CognitiveDynamics cog1, cog2;

    brain1.initialize();
    brain2.initialize();

    // Build identical graphs
    auto c1_1 = ltm1.store_concept("A", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c1_2 = ltm1.store_concept("B", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.8));
    ltm1.add_relation(c1_1, c1_2, RelationType::IS_A, 0.7);

    auto c2_1 = ltm2.store_concept("A", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c2_2 = ltm2.store_concept("B", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.8));
    ltm2.add_relation(c2_1, c2_2, RelationType::IS_A, 0.7);

    // Spread activation
    ContextId ctx1 = brain1.create_context();
    ContextId ctx2 = brain2.create_context();

    ShortTermMemory* stm1 = brain1.get_stm_mutable();
    ShortTermMemory* stm2 = brain2.get_stm_mutable();

    auto stats1 = cog1.spread_activation(c1_1, 1.0, ctx1, ltm1, *stm1);
    auto stats2 = cog2.spread_activation(c2_1, 1.0, ctx2, ltm2, *stm2);

    // Results MUST be identical
    assert(stats1.concepts_activated == stats2.concepts_activated);
    assert(stats1.max_depth_reached == stats2.max_depth_reached);
    assert(std::abs(stats1.total_activation_added - stats2.total_activation_added) < 0.0001);

    double act1_1 = stm1->get_concept_activation(ctx1, c1_1);
    double act2_1 = stm2->get_concept_activation(ctx2, c2_1);
    assert(std::abs(act1_1 - act2_1) < 0.0001);

    test_pass("Same inputs produce same outputs");
    test_pass("Spreading is deterministic");

    brain1.destroy_context(ctx1);
    brain2.destroy_context(ctx2);
    brain1.shutdown();
    brain2.shutdown();
}

// =============================================================================
// TEST 3: Bounded Activations
// =============================================================================
void test_bounded_activations() {
    print_test_header("Bounded Activation Values");

    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;

    brain.initialize();

    // Create chain with high weights
    auto a = ltm.store_concept("A", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 1.0));
    auto b = ltm.store_concept("B", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 1.0));
    auto c = ltm.store_concept("C", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 1.0));

    ltm.add_relation(a, b, RelationType::IS_A, 1.0);
    ltm.add_relation(b, c, RelationType::IS_A, 1.0);

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    cognitive.spread_activation(a, 1.0, ctx, ltm, *stm);

    // Check all activations are bounded [0.0, 1.0]
    double act_a = stm->get_concept_activation(ctx, a);
    double act_b = stm->get_concept_activation(ctx, b);
    double act_c = stm->get_concept_activation(ctx, c);

    assert(act_a >= 0.0 && act_a <= 1.0);
    assert(act_b >= 0.0 && act_b <= 1.0);
    assert(act_c >= 0.0 && act_c <= 1.0);

    test_pass("All activations in [0.0, 1.0]");

    brain.destroy_context(ctx);
    brain.shutdown();
}

// =============================================================================
// TEST 4: Cycle Detection
// =============================================================================
void test_cycle_detection() {
    print_test_header("Cycle Detection");

    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;

    brain.initialize();

    // Create cycle: A -> B -> C -> A
    auto a = ltm.store_concept("A", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto b = ltm.store_concept("B", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c = ltm.store_concept("C", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    ltm.add_relation(a, b, RelationType::IS_A, 0.8);
    ltm.add_relation(b, c, RelationType::IS_A, 0.8);
    ltm.add_relation(c, a, RelationType::IS_A, 0.8);  // Cycle!

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    // Should NOT hang (cycle detection prevents infinite loop)
    auto stats = cognitive.spread_activation(a, 1.0, ctx, ltm, *stm);

    // Should terminate and activate all nodes
    assert(stats.concepts_activated == 3);

    test_pass("Cycle detected and handled");
    test_pass("No infinite loop");

    brain.destroy_context(ctx);
    brain.shutdown();
}

// =============================================================================
// TEST 5: Focus Decay
// =============================================================================
void test_focus_decay() {
    print_test_header("Focus Decay");

    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;

    brain.initialize();

    auto cat = ltm.store_concept("Cat", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    ContextId ctx = brain.create_context();
    cognitive.init_focus(ctx);

    // Focus on concept
    cognitive.focus_on(ctx, cat, 0.0);

    double initial_focus = cognitive.get_focus_score(ctx, cat);
    assert(initial_focus > 0.0);

    // Apply decay
    cognitive.decay_focus(ctx);
    cognitive.decay_focus(ctx);

    double decayed_focus = cognitive.get_focus_score(ctx, cat);

    // Focus should have decreased
    assert(decayed_focus < initial_focus);

    test_pass("Focus decays over time");

    brain.destroy_context(ctx);
    brain.shutdown();
}

// =============================================================================
// TEST 6: Salience Ranking
// =============================================================================
void test_salience_ranking() {
    print_test_header("Salience Ranking");

    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;

    brain.initialize();

    auto high_trust = ltm.store_concept("High", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    auto low_trust = ltm.store_concept("Low", "Test",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.30));

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    // Activate both equally
    stm->activate_concept(ctx, high_trust, 0.8, ActivationClass::CORE_KNOWLEDGE);
    stm->activate_concept(ctx, low_trust, 0.8, ActivationClass::CORE_KNOWLEDGE);

    // Compute salience
    std::vector<ConceptId> concepts = {high_trust, low_trust};
    auto scores = cognitive.compute_salience_batch(concepts, ctx, ltm, *stm, 0);

    // High trust should have higher salience
    assert(scores.size() == 2);
    assert(scores[0].salience > scores[1].salience);

    test_pass("Salience ranking correct");
    test_pass("Trust affects salience");

    brain.destroy_context(ctx);
    brain.shutdown();
}

// =============================================================================
// TEST 7: Path Finding
// =============================================================================
void test_path_finding() {
    print_test_header("Thought Path Finding");

    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;

    brain.initialize();

    auto a = ltm.store_concept("A", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto b = ltm.store_concept("B", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto c = ltm.store_concept("C", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    ltm.add_relation(a, b, RelationType::IS_A, 0.9);
    ltm.add_relation(b, c, RelationType::IS_A, 0.9);

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    // Activate concepts
    cognitive.spread_activation(a, 1.0, ctx, ltm, *stm);

    // Find paths from A to C
    auto paths = cognitive.find_paths_to(a, c, ctx, ltm, *stm);

    // Should find at least one path
    assert(!paths.empty());

    // First path should be A -> B -> C
    assert(paths[0].nodes.size() == 3);
    assert(paths[0].nodes[0].concept_id == a);
    assert(paths[0].nodes[1].concept_id == b);
    assert(paths[0].nodes[2].concept_id == c);

    test_pass("Thought paths found");
    test_pass("Path ranking works");

    brain.destroy_context(ctx);
    brain.shutdown();
}

// =============================================================================
// TEST 8: INVALIDATED Concepts
// =============================================================================
void test_invalidated_concepts() {
    print_test_header("INVALIDATED Concepts Not Propagated");

    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;

    brain.initialize();

    auto valid = ltm.store_concept("Valid", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto invalid = ltm.store_concept("Invalid", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
    auto target = ltm.store_concept("Target", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));

    ltm.add_relation(valid, target, RelationType::IS_A, 0.9);
    ltm.add_relation(invalid, target, RelationType::IS_A, 0.9);

    // Invalidate one concept
    ltm.invalidate_concept(invalid);

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    // Try to spread from invalidated concept
    auto stats = cognitive.spread_activation(invalid, 1.0, ctx, ltm, *stm);

    // INVALIDATED concept should not propagate
    // Only the source itself should be activated, target should NOT be
    double target_activation = stm->get_concept_activation(ctx, target);
    assert(target_activation == 0.0);  // Should NOT be activated

    test_pass("INVALIDATED concepts do not propagate");

    brain.destroy_context(ctx);
    brain.shutdown();
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Brain19 - Cognitive Dynamics Test Suite            ║\n";
    std::cout << "║  Erweiterte Tests (8 Tests)                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    try {
        test_epistemic_invariants();
        test_spreading_determinism();
        test_bounded_activations();
        test_cycle_detection();
        test_focus_decay();
        test_salience_ranking();
        test_path_finding();
        test_invalidated_concepts();

    } catch (const std::exception& e) {
        std::cout << "\n✗ EXCEPTION: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Total Tests: " << test_count << "\n";
    std::cout << "Assertions Passed: " << passed_count << "\n";
    std::cout << "\n✅ ALL TESTS PASSED!\n\n";

    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  COGNITIVE DYNAMICS IMPLEMENTATION - VERIFIED ✓      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    return 0;
}
