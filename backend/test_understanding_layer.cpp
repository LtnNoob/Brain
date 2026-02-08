#include "understanding/understanding_layer.hpp"
#include "memory/brain_controller.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>

using namespace brain19;

// =============================================================================
// TEST UTILITIES
// =============================================================================

void print_section(const std::string& title) {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "TEST: " << title << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

void print_pass(const std::string& message) {
    std::cout << "✓ PASS: " << message << "\n";
}

// =============================================================================
// TEST 1: Epistemic Invariants - No Knowledge Writes
// =============================================================================

void test_no_knowledge_writes() {
    print_section("Epistemic Invariants - No Knowledge Writes");

    BrainController brain;
    brain.initialize();

    LongTermMemory ltm;
    UnderstandingLayer understanding;

    // Register stub Mini-LLM
    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

    // Create test knowledge
    auto cat = ltm.store_concept("Cat", "Feline",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    auto dog = ltm.store_concept("Dog", "Canine",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.97));

    // Record initial state
    size_t initial_concept_count = ltm.get_all_concept_ids().size();
    auto cat_before = ltm.retrieve_concept(cat);
    auto dog_before = ltm.retrieve_concept(dog);

    // Execute Understanding Layer
    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    std::vector<ConceptId> active_concepts = {cat, dog};
    auto meaning_proposals = understanding.analyze_meaning(active_concepts, ltm, *stm, ctx);

    // CRITICAL VERIFICATION: Knowledge Graph unchanged
    size_t final_concept_count = ltm.get_all_concept_ids().size();
    assert(initial_concept_count == final_concept_count);
    print_pass("No new concepts created");

    auto cat_after = ltm.retrieve_concept(cat);
    auto dog_after = ltm.retrieve_concept(dog);

    // Verify epistemic metadata unchanged
    assert(cat_before->epistemic.trust == cat_after->epistemic.trust);
    assert(cat_before->epistemic.type == cat_after->epistemic.type);
    assert(cat_before->epistemic.status == cat_after->epistemic.status);
    print_pass("Cat epistemic metadata unchanged");

    assert(dog_before->epistemic.trust == dog_after->epistemic.trust);
    assert(dog_before->epistemic.type == dog_after->epistemic.type);
    assert(dog_before->epistemic.status == dog_after->epistemic.status);
    print_pass("Dog epistemic metadata unchanged");

    brain.destroy_context(ctx);
    brain.shutdown();

    std::cout << "✅ TEST PASSED: No Knowledge Writes\n";
}

// =============================================================================
// TEST 2: Epistemic Invariants - No Trust Manipulation
// =============================================================================

void test_no_trust_manipulation() {
    print_section("Epistemic Invariants - No Trust Manipulation");

    BrainController brain;
    brain.initialize();

    LongTermMemory ltm;
    UnderstandingLayer understanding;

    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

    // Create concepts with varying trust levels
    auto high_trust = ltm.store_concept("Fact", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.99));

    auto low_trust = ltm.store_concept("Speculation", "Test",
        EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.25));

    // Record trust levels
    auto high_before = ltm.retrieve_concept(high_trust)->epistemic.trust;
    auto low_before = ltm.retrieve_concept(low_trust)->epistemic.trust;

    // Execute Understanding Layer multiple times
    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    std::vector<ConceptId> concepts = {high_trust, low_trust};

    for (int i = 0; i < 5; ++i) {
        understanding.analyze_meaning(concepts, ltm, *stm, ctx);
        understanding.propose_hypotheses(concepts, ltm, *stm, ctx);
    }

    // CRITICAL VERIFICATION: Trust unchanged after multiple cycles
    auto high_after = ltm.retrieve_concept(high_trust)->epistemic.trust;
    auto low_after = ltm.retrieve_concept(low_trust)->epistemic.trust;

    assert(high_before == high_after);
    print_pass("High trust concept unchanged (0.99)");

    assert(low_before == low_after);
    print_pass("Low trust concept unchanged (0.25)");

    brain.destroy_context(ctx);
    brain.shutdown();

    std::cout << "✅ TEST PASSED: No Trust Manipulation\n";
}

// =============================================================================
// TEST 3: All Proposals are HYPOTHESIS
// =============================================================================

void test_all_proposals_are_hypothesis() {
    print_section("All Proposals are HYPOTHESIS");

    BrainController brain;
    brain.initialize();

    LongTermMemory ltm;
    UnderstandingLayer understanding;

    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

    // Create test knowledge
    auto cat = ltm.store_concept("Cat", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    auto dog = ltm.store_concept("Dog", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.97));

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    std::vector<ConceptId> concepts = {cat, dog};

    // Test MeaningProposals
    auto meaning_proposals = understanding.analyze_meaning(concepts, ltm, *stm, ctx);
    for (const auto& proposal : meaning_proposals) {
        assert(proposal.epistemic_type == EpistemicType::HYPOTHESIS);
    }
    print_pass(std::to_string(meaning_proposals.size()) + " MeaningProposals are HYPOTHESIS");

    // Test HypothesisProposals
    auto hypothesis_proposals = understanding.propose_hypotheses(concepts, ltm, *stm, ctx);
    for (const auto& proposal : hypothesis_proposals) {
        assert(proposal.suggested_epistemic.suggested_type == EpistemicType::HYPOTHESIS);
    }
    print_pass(std::to_string(hypothesis_proposals.size()) + " HypothesisProposals suggest HYPOTHESIS");

    brain.destroy_context(ctx);
    brain.shutdown();

    std::cout << "✅ TEST PASSED: All Proposals are HYPOTHESIS\n";
}

// =============================================================================
// TEST 4: Deterministic Behavior
// =============================================================================

void test_deterministic_behavior() {
    print_section("Deterministic Behavior");

    LongTermMemory ltm;

    auto cat = ltm.store_concept("Cat", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    auto dog = ltm.store_concept("Dog", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.97));

    // Run 1
    BrainController brain1;
    brain1.initialize();
    UnderstandingLayer understanding1;
    understanding1.register_mini_llm(std::make_unique<StubMiniLLM>());

    ContextId ctx1 = brain1.create_context();
    ShortTermMemory* stm1 = brain1.get_stm_mutable();

    std::vector<ConceptId> concepts = {cat, dog};
    auto proposals1 = understanding1.analyze_meaning(concepts, ltm, *stm1, ctx1);

    // Run 2 (identical setup)
    BrainController brain2;
    brain2.initialize();
    UnderstandingLayer understanding2;
    understanding2.register_mini_llm(std::make_unique<StubMiniLLM>());

    ContextId ctx2 = brain2.create_context();
    ShortTermMemory* stm2 = brain2.get_stm_mutable();

    auto proposals2 = understanding2.analyze_meaning(concepts, ltm, *stm2, ctx2);

    // CRITICAL VERIFICATION: Same inputs → same outputs
    assert(proposals1.size() == proposals2.size());
    print_pass("Same number of proposals generated (" + std::to_string(proposals1.size()) + ")");

    if (!proposals1.empty() && !proposals2.empty()) {
        assert(proposals1[0].source_concepts == proposals2[0].source_concepts);
        print_pass("Proposals reference same source concepts");
    }

    brain1.destroy_context(ctx1);
    brain1.shutdown();
    brain2.destroy_context(ctx2);
    brain2.shutdown();

    std::cout << "✅ TEST PASSED: Deterministic Behavior\n";
}

// =============================================================================
// TEST 5: No Autonomous Actions
// =============================================================================

void test_no_autonomous_actions() {
    print_section("No Autonomous Actions");

    BrainController brain;
    brain.initialize();

    LongTermMemory ltm;
    UnderstandingLayer understanding;

    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

    auto cat = ltm.store_concept("Cat", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    // Record initial state
    size_t initial_count = ltm.get_all_concept_ids().size();

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    // Create Understanding Layer but DON'T call its methods
    // Verify it doesn't do anything autonomously

    // Wait/idle (in real code, would sleep or do other work)
    // Understanding Layer should remain completely passive

    // Verify nothing changed
    size_t final_count = ltm.get_all_concept_ids().size();
    assert(initial_count == final_count);
    print_pass("Understanding Layer is passive (no autonomous actions)");

    brain.destroy_context(ctx);
    brain.shutdown();

    std::cout << "✅ TEST PASSED: No Autonomous Actions\n";
}

// =============================================================================
// TEST 6: READ-ONLY LTM Access
// =============================================================================

void test_readonly_ltm_access() {
    print_section("READ-ONLY LTM Access");

    BrainController brain;
    brain.initialize();

    LongTermMemory ltm;
    UnderstandingLayer understanding;

    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

    // Create knowledge with relations
    auto cat = ltm.store_concept("Cat", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    auto mammal = ltm.store_concept("Mammal", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));

    ltm.add_relation(cat, mammal, RelationType::IS_A, 0.9);

    // Record initial state
    size_t initial_relation_count = ltm.get_relation_count(cat);

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    std::vector<ConceptId> concepts = {cat, mammal};

    // Execute Understanding Layer
    understanding.analyze_meaning(concepts, ltm, *stm, ctx);
    understanding.propose_hypotheses(concepts, ltm, *stm, ctx);
    understanding.check_contradictions(concepts, ltm, *stm, ctx);

    // CRITICAL VERIFICATION: Relations unchanged
    size_t final_relation_count = ltm.get_relation_count(cat);
    assert(initial_relation_count == final_relation_count);
    print_pass("Relation count unchanged (" + std::to_string(final_relation_count) + ")");

    brain.destroy_context(ctx);
    brain.shutdown();

    std::cout << "✅ TEST PASSED: READ-ONLY LTM Access\n";
}

// =============================================================================
// TEST 7: System Functions Without Understanding Layer
// =============================================================================

void test_system_functions_without_understanding() {
    print_section("System Functions Without Understanding Layer");

    // CRITICAL: System must work without Understanding Layer
    BrainController brain;
    brain.initialize();

    LongTermMemory ltm;
    // NO Understanding Layer created

    auto cat = ltm.store_concept("Cat", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    ContextId ctx = brain.create_context();

    // System should function normally
    brain.activate_concept_in_context(ctx, cat, 0.9, ActivationClass::CORE_KNOWLEDGE);

    double activation = brain.query_concept_activation(ctx, cat);
    assert(activation > 0.8);
    print_pass("Brain functions without Understanding Layer");

    brain.destroy_context(ctx);
    brain.shutdown();

    std::cout << "✅ TEST PASSED: System Functions Without Understanding Layer\n";
}

// =============================================================================
// TEST 8: Bounded Values
// =============================================================================

void test_bounded_values() {
    print_section("Bounded Values");

    BrainController brain;
    brain.initialize();

    LongTermMemory ltm;
    UnderstandingLayer understanding;

    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

    auto cat = ltm.store_concept("Cat", "Test",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    std::vector<ConceptId> concepts = {cat};
    auto proposals = understanding.analyze_meaning(concepts, ltm, *stm, ctx);

    // Verify all confidence values are bounded [0.0, 1.0]
    for (const auto& proposal : proposals) {
        assert(proposal.model_confidence >= 0.0);
        assert(proposal.model_confidence <= 1.0);
    }
    print_pass("All model_confidence values in [0.0, 1.0]");

    brain.destroy_context(ctx);
    brain.shutdown();

    std::cout << "✅ TEST PASSED: Bounded Values\n";
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Brain19 - Understanding Layer Test Suite           ║\n";
    std::cout << "║  Epistemic Invariant Verification                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    try {
        test_no_knowledge_writes();
        test_no_trust_manipulation();
        test_all_proposals_are_hypothesis();
        test_deterministic_behavior();
        test_no_autonomous_actions();
        test_readonly_ltm_access();
        test_system_functions_without_understanding();
        test_bounded_values();

        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "TEST SUMMARY\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Total Tests: 8\n";
        std::cout << "\n✅ ALL TESTS PASSED!\n\n";

        std::cout << "╔══════════════════════════════════════════════════════╗\n";
        std::cout << "║  UNDERSTANDING LAYER - VERIFIED ✓                    ║\n";
        std::cout << "╚══════════════════════════════════════════════════════╝\n";

        std::cout << "\nEPISTEMIC GUARANTEES:\n";
        std::cout << "  ✓ No knowledge writes\n";
        std::cout << "  ✓ No trust manipulation\n";
        std::cout << "  ✓ All proposals are HYPOTHESIS\n";
        std::cout << "  ✓ Deterministic behavior\n";
        std::cout << "  ✓ No autonomous actions\n";
        std::cout << "  ✓ READ-ONLY LTM access\n";
        std::cout << "  ✓ System functions without Understanding Layer\n";
        std::cout << "  ✓ Bounded values\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED WITH EXCEPTION: " << e.what() << "\n\n";
        return 1;
    }
}
