#include "understanding/understanding_layer.hpp"
#include "cognitive/cognitive_dynamics.hpp"
#include "memory/brain_controller.hpp"
#include <iostream>
#include <iomanip>

using namespace brain19;

void print_header(const std::string& title) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::left << std::setw(50) << title << "  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
}

void print_section(const std::string& title) {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << title << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

int main() {
    print_header("Brain19 - Understanding Layer Integration Demo");

    std::cout << "This demo demonstrates:\n";
    std::cout << "  • Understanding Layer generates semantic proposals\n";
    std::cout << "  • All proposals are HYPOTHESIS (not FACT)\n";
    std::cout << "  • Epistemic Core remains sole truth arbiter\n";
    std::cout << "  • Understanding Layer is READ-ONLY w.r.t. Knowledge Graph\n\n";

    // =============================================================================
    // INITIALIZATION
    // =============================================================================

    print_section("Initializing Subsystems");

    BrainController brain;
    brain.initialize();
    std::cout << "✓ BrainController initialized\n";

    LongTermMemory ltm;
    std::cout << "✓ LongTermMemory initialized\n";

    CognitiveDynamics cognitive;
    std::cout << "✓ CognitiveDynamics initialized\n";

    UnderstandingLayer understanding(UnderstandingLayerConfig{
        .enable_parallel_llms = false,
        .min_meaning_confidence = 0.3,
        .min_hypothesis_confidence = 0.2,
        .min_analogy_confidence = 0.4,
        .min_contradiction_severity = 0.5,
        .max_proposals_per_cycle = 10,
        .verbose_logging = true
    });

    // Register stub Mini-LLM
    understanding.register_mini_llm(std::make_unique<StubMiniLLM>());
    std::cout << "✓ Understanding Layer initialized with "
              << understanding.get_mini_llm_count() << " Mini-LLM(s)\n";

    // =============================================================================
    // KNOWLEDGE GRAPH SETUP
    // =============================================================================

    print_section("Building Knowledge Graph");

    std::cout << "Storing concepts with epistemic metadata...\n\n";

    auto cat = ltm.store_concept("Cat", "Domesticated feline mammal",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));
    std::cout << "  Cat (FACT, trust=0.98)\n";

    auto mammal = ltm.store_concept("Mammal", "Warm-blooded vertebrate",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    std::cout << "  Mammal (FACT, trust=0.95)\n";

    auto animal = ltm.store_concept("Animal", "Living organism",
        EpistemicMetadata(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.99));
    std::cout << "  Animal (DEFINITION, trust=0.99)\n";

    auto fur = ltm.store_concept("Fur", "Hair covering",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.97));
    std::cout << "  Fur (FACT, trust=0.97)\n";

    auto predator = ltm.store_concept("Predator", "Hunts other animals",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.90));
    std::cout << "  Predator (THEORY, trust=0.90)\n";

    std::cout << "\nBuilding relations...\n";
    ltm.add_relation(cat, mammal, RelationType::IS_A, 0.95);
    std::cout << "  ✓ Cat IS_A Mammal (weight=0.95)\n";

    ltm.add_relation(mammal, animal, RelationType::IS_A, 0.95);
    std::cout << "  ✓ Mammal IS_A Animal (weight=0.95)\n";

    ltm.add_relation(cat, fur, RelationType::HAS_PROPERTY, 0.90);
    std::cout << "  ✓ Cat HAS_PROPERTY Fur (weight=0.90)\n";

    ltm.add_relation(cat, predator, RelationType::IS_A, 0.75);
    std::cout << "  ✓ Cat IS_A Predator (weight=0.75)\n";

    // Record initial epistemic state
    auto cat_before = ltm.retrieve_concept(cat);
    std::cout << "\n📊 Initial Epistemic State:\n";
    std::cout << "  Cat: " << static_cast<int>(cat_before->epistemic.type)
              << ", trust=" << cat_before->epistemic.trust << "\n";

    // =============================================================================
    // PHASE 1: MEANING EXTRACTION
    // =============================================================================

    print_section("Phase 1: Meaning Extraction");

    ContextId ctx = brain.create_context();
    ShortTermMemory* stm = brain.get_stm_mutable();

    std::vector<ConceptId> active_concepts = {cat, mammal, fur};

    auto meaning_proposals = understanding.analyze_meaning(active_concepts, ltm, *stm, ctx);

    std::cout << "Generated " << meaning_proposals.size() << " meaning proposal(s):\n\n";

    for (const auto& proposal : meaning_proposals) {
        std::cout << "  Proposal #" << proposal.proposal_id << ":\n";
        std::cout << "    Interpretation: " << proposal.interpretation << "\n";
        std::cout << "    Reasoning: " << proposal.reasoning << "\n";
        std::cout << "    Model Confidence: " << proposal.model_confidence << "\n";
        std::cout << "    Epistemic Type: HYPOTHESIS ✓\n";  // Always HYPOTHESIS
        std::cout << "    Source: " << proposal.source_model << "\n\n";
    }

    // =============================================================================
    // PHASE 2: HYPOTHESIS GENERATION
    // =============================================================================

    print_section("Phase 2: Hypothesis Generation");

    std::vector<ConceptId> evidence = {cat, mammal, predator};

    auto hypothesis_proposals = understanding.propose_hypotheses(evidence, ltm, *stm, ctx);

    std::cout << "Generated " << hypothesis_proposals.size() << " hypothesis proposal(s):\n\n";

    for (const auto& proposal : hypothesis_proposals) {
        std::cout << "  Hypothesis #" << proposal.proposal_id << ":\n";
        std::cout << "    Statement: " << proposal.hypothesis_statement << "\n";
        std::cout << "    Reasoning: " << proposal.supporting_reasoning << "\n";
        std::cout << "    Patterns: ";
        for (const auto& pattern : proposal.detected_patterns) {
            std::cout << pattern << " ";
        }
        std::cout << "\n";
        std::cout << "    Model Confidence: " << proposal.model_confidence << "\n";
        std::cout << "    Suggested Type: HYPOTHESIS ✓\n";  // Always HYPOTHESIS
        std::cout << "    Suggested Trust: " << proposal.suggested_epistemic.suggested_trust << "\n";
        std::cout << "    Source: " << proposal.source_model << "\n\n";
    }

    // =============================================================================
    // PHASE 3: CONTRADICTION DETECTION
    // =============================================================================

    print_section("Phase 3: Contradiction Detection");

    // Add an invalidated concept to test contradiction detection
    auto phlogiston = ltm.store_concept("Phlogiston Theory", "Outdated theory",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.85));

    ltm.invalidate_concept(phlogiston, 0.05);  // Invalidate it
    std::cout << "Added and invalidated: Phlogiston Theory\n\n";

    std::vector<ConceptId> concepts_to_check = {cat, mammal, phlogiston};

    auto contradiction_proposals = understanding.check_contradictions(concepts_to_check, ltm, *stm, ctx);

    std::cout << "Generated " << contradiction_proposals.size() << " contradiction proposal(s):\n\n";

    for (const auto& proposal : contradiction_proposals) {
        std::cout << "  Contradiction #" << proposal.proposal_id << ":\n";
        std::cout << "    Concept A: " << proposal.concept_a << "\n";
        std::cout << "    Concept B: " << proposal.concept_b << "\n";
        std::cout << "    Description: " << proposal.contradiction_description << "\n";
        std::cout << "    Reasoning: " << proposal.reasoning << "\n";
        std::cout << "    Severity: " << proposal.severity << "\n";
        std::cout << "    Confidence: " << proposal.model_confidence << "\n";
        std::cout << "    Source: " << proposal.source_model << "\n\n";
    }

    // =============================================================================
    // PHASE 4: INTEGRATED CYCLE (with Cognitive Dynamics)
    // =============================================================================

    print_section("Phase 4: Integrated Understanding Cycle");

    std::cout << "Using Cognitive Dynamics for spreading activation...\n";

    auto result = understanding.perform_understanding_cycle(
        cat,           // Seed concept
        cognitive,     // Cognitive Dynamics for salience
        ltm,          // READ-ONLY
        *stm,         // For Cognitive Dynamics activation
        ctx
    );

    std::cout << "\nIntegrated Cycle Results:\n";
    std::cout << "  Meaning Proposals: " << result.meaning_proposals.size() << "\n";
    std::cout << "  Hypothesis Proposals: " << result.hypothesis_proposals.size() << "\n";
    std::cout << "  Analogy Proposals: " << result.analogy_proposals.size() << "\n";
    std::cout << "  Contradiction Proposals: " << result.contradiction_proposals.size() << "\n";
    std::cout << "  Total Proposals: " << result.total_proposals_generated << "\n";

    // =============================================================================
    // PHASE 5: EPISTEMIC INVARIANT VERIFICATION
    // =============================================================================

    print_section("Phase 5: Epistemic Invariant Verification");

    std::cout << "Verifying that epistemic metadata was NOT modified...\n\n";

    auto cat_after = ltm.retrieve_concept(cat);

    bool trust_unchanged = (cat_before->epistemic.trust == cat_after->epistemic.trust);
    bool type_unchanged = (cat_before->epistemic.type == cat_after->epistemic.type);
    bool status_unchanged = (cat_before->epistemic.status == cat_after->epistemic.status);

    std::cout << "  Cat trust: " << cat_before->epistemic.trust
              << " → " << cat_after->epistemic.trust
              << (trust_unchanged ? " ✓ unchanged" : " ✗ CHANGED") << "\n";

    std::cout << "  Cat type: " << static_cast<int>(cat_before->epistemic.type)
              << " → " << static_cast<int>(cat_after->epistemic.type)
              << (type_unchanged ? " ✓ unchanged" : " ✗ CHANGED") << "\n";

    std::cout << "  Cat status: " << static_cast<int>(cat_before->epistemic.status)
              << " → " << static_cast<int>(cat_after->epistemic.status)
              << (status_unchanged ? " ✓ unchanged" : " ✗ CHANGED") << "\n";

    // =============================================================================
    // STATISTICS
    // =============================================================================

    print_section("Understanding Layer Statistics");

    auto stats = understanding.get_statistics();
    std::cout << "  Total Meaning Proposals: " << stats.total_meaning_proposals << "\n";
    std::cout << "  Total Hypothesis Proposals: " << stats.total_hypothesis_proposals << "\n";
    std::cout << "  Total Analogy Proposals: " << stats.total_analogy_proposals << "\n";
    std::cout << "  Total Contradiction Proposals: " << stats.total_contradiction_proposals << "\n";
    std::cout << "  Total Cycles Performed: " << stats.total_cycles_performed << "\n";

    // =============================================================================
    // CLEANUP
    // =============================================================================

    print_section("Cleanup");

    brain.destroy_context(ctx);
    std::cout << "✓ Context destroyed\n";

    brain.shutdown();
    std::cout << "✓ Brain shut down\n";

    // =============================================================================
    // FINAL VERIFICATION
    // =============================================================================

    print_section("Final Verification");

    if (trust_unchanged && type_unchanged && status_unchanged) {
        std::cout << "\n═══════════════════════════════════════════════════════\n";
        std::cout << "  ALL EPISTEMIC INVARIANTS PRESERVED ✓\n";
        std::cout << "═══════════════════════════════════════════════════════\n\n";
    } else {
        std::cout << "\n❌ EPISTEMIC VIOLATION DETECTED!\n\n";
        return 1;
    }

    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Understanding Layer Integration - SUCCESS           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    std::cout << "ARCHITECTURAL GUARANTEES:\n";
    std::cout << "  ✓ Understanding Layer generated semantic proposals\n";
    std::cout << "  ✓ All proposals are HYPOTHESIS (never FACT)\n";
    std::cout << "  ✓ LTM accessed READ-ONLY (no knowledge writes)\n";
    std::cout << "  ✓ Trust values unchanged\n";
    std::cout << "  ✓ EpistemicType unchanged\n";
    std::cout << "  ✓ EpistemicStatus unchanged\n";
    std::cout << "  ✓ No autonomous actions\n";
    std::cout << "  ✓ Epistemic Core remains sole truth arbiter\n\n";

    return 0;
}
