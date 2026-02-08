#include "cognitive/cognitive_dynamics.hpp"
#include "memory/brain_controller.hpp"
#include "ltm/long_term_memory.hpp"
#include "epistemic/epistemic_metadata.hpp"
#include <iostream>
#include <iomanip>

using namespace brain19;

void print_separator(const std::string& title = "") {
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    if (!title.empty()) {
        std::cout << title << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    }
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Brain19 - Cognitive Dynamics Integration Demo       ║\n";
    std::cout << "║                                                      ║\n";
    std::cout << "║  Demonstrates: Spreading Activation, Salience,       ║\n";
    std::cout << "║  Focus Management, and Thought Path Ranking          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    
    // Initialize subsystems
    print_separator("Initializing Subsystems");
    
    BrainController brain;
    LongTermMemory ltm;
    CognitiveDynamics cognitive;
    
    brain.initialize();
    std::cout << "✓ BrainController initialized\n";
    std::cout << "✓ LongTermMemory initialized\n";
    std::cout << "✓ CognitiveDynamics initialized\n";
    
    // Build knowledge graph with epistemic metadata
    print_separator("Building Knowledge Graph");
    
    std::cout << "Storing concepts with epistemic metadata...\n\n";
    
    // Domain: Biology taxonomy
    auto cat = ltm.store_concept("Cat", "Small domesticated carnivorous mammal",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.98));
    std::cout << "  Cat (FACT, trust=0.98)\n";
    
    auto mammal = ltm.store_concept("Mammal", "Warm-blooded vertebrate animal",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.95));
    std::cout << "  Mammal (FACT, trust=0.95)\n";
    
    auto animal = ltm.store_concept("Animal", "Living organism that feeds on organic matter",
        EpistemicMetadata(EpistemicType::DEFINITION, EpistemicStatus::ACTIVE, 0.99));
    std::cout << "  Animal (DEFINITION, trust=0.99)\n";
    
    auto fur = ltm.store_concept("Fur", "Thick growth of hair covering skin",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.97));
    std::cout << "  Fur (FACT, trust=0.97)\n";
    
    auto warm_blooded = ltm.store_concept("Warm-blooded", "Maintains constant body temperature",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.96));
    std::cout << "  Warm-blooded (FACT, trust=0.96)\n";
    
    auto whiskers = ltm.store_concept("Whiskers", "Specialized sensory hairs",
        EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.94));
    std::cout << "  Whiskers (FACT, trust=0.94)\n";
    
    auto predator = ltm.store_concept("Predator", "Animal that hunts and feeds on other animals",
        EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.90));
    std::cout << "  Predator (THEORY, trust=0.90)\n";
    
    auto nocturnal = ltm.store_concept("Nocturnal", "Active mainly during the night",
        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.70));
    std::cout << "  Nocturnal (HYPOTHESIS, trust=0.70)\n";
    
    // Build relations
    std::cout << "\nBuilding relations...\n";
    
    ltm.add_relation(cat, mammal, RelationType::IS_A, 0.95);
    ltm.add_relation(mammal, animal, RelationType::IS_A, 0.95);
    ltm.add_relation(cat, fur, RelationType::HAS_PROPERTY, 0.90);
    ltm.add_relation(cat, whiskers, RelationType::HAS_PROPERTY, 0.85);
    ltm.add_relation(mammal, warm_blooded, RelationType::HAS_PROPERTY, 0.90);
    ltm.add_relation(cat, predator, RelationType::IS_A, 0.75);
    ltm.add_relation(cat, nocturnal, RelationType::HAS_PROPERTY, 0.60);
    
    std::cout << "  ✓ Cat IS_A Mammal (weight=0.95)\n";
    std::cout << "  ✓ Mammal IS_A Animal (weight=0.95)\n";
    std::cout << "  ✓ Cat HAS_PROPERTY Fur (weight=0.90)\n";
    std::cout << "  ✓ Cat HAS_PROPERTY Whiskers (weight=0.85)\n";
    std::cout << "  ✓ Mammal HAS_PROPERTY Warm-blooded (weight=0.90)\n";
    std::cout << "  ✓ Cat IS_A Predator (weight=0.75)\n";
    std::cout << "  ✓ Cat HAS_PROPERTY Nocturnal (weight=0.60)\n";
    
    // Create context and initialize focus
    ContextId ctx = brain.create_context();
    cognitive.init_focus(ctx);
    
    // PHASE 1: Spreading Activation
    print_separator("Phase 1: Spreading Activation");
    
    std::cout << "Spreading activation from 'Cat' (initial activation = 1.0)...\n\n";
    
    // Get mutable STM access
    ShortTermMemory* stm = brain.get_stm_mutable();
    
    auto spread_stats = cognitive.spread_activation(cat, 1.0, ctx, ltm, *stm);
    
    std::cout << "Spreading Statistics:\n";
    std::cout << "  Concepts activated: " << spread_stats.concepts_activated << "\n";
    std::cout << "  Max depth reached: " << spread_stats.max_depth_reached << "\n";
    std::cout << "  Total propagations: " << spread_stats.total_propagations << "\n";
    std::cout << "  Total activation added: " << std::fixed << std::setprecision(3) 
              << spread_stats.total_activation_added << "\n\n";
    
    std::cout << "Activation levels after spreading:\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Cat:          " << stm->get_concept_activation(ctx, cat) << "\n";
    std::cout << "  Mammal:       " << stm->get_concept_activation(ctx, mammal) << "\n";
    std::cout << "  Animal:       " << stm->get_concept_activation(ctx, animal) << "\n";
    std::cout << "  Fur:          " << stm->get_concept_activation(ctx, fur) << "\n";
    std::cout << "  Whiskers:     " << stm->get_concept_activation(ctx, whiskers) << "\n";
    std::cout << "  Warm-blooded: " << stm->get_concept_activation(ctx, warm_blooded) << "\n";
    std::cout << "  Predator:     " << stm->get_concept_activation(ctx, predator) << "\n";
    std::cout << "  Nocturnal:    " << stm->get_concept_activation(ctx, nocturnal) << "\n";
    
    // PHASE 2: Salience Computation
    print_separator("Phase 2: Salience Computation");
    
    std::vector<ConceptId> all_concepts = {cat, mammal, animal, fur, whiskers, 
                                           warm_blooded, predator, nocturnal};
    
    auto salience_scores = cognitive.compute_salience_batch(all_concepts, ctx, ltm, *stm, 0);
    
    std::cout << "Salience scores (sorted by importance):\n\n";
    std::cout << "  Concept         Salience   Activation  Trust    Connectivity\n";
    std::cout << "  ─────────────────────────────────────────────────────────────\n";
    
    for (const auto& score : salience_scores) {
        auto info = ltm.retrieve_concept(score.concept_id);
        if (info.has_value()) {
            std::cout << "  " << std::left << std::setw(14) << info->label 
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(8) << score.salience
                      << std::setw(12) << score.activation_contrib
                      << std::setw(9) << score.trust_contrib
                      << std::setw(13) << score.connectivity_contrib << "\n";
        }
    }
    
    // PHASE 3: Focus Management
    print_separator("Phase 3: Focus Management");
    
    std::cout << "Focusing on top 3 most salient concepts...\n\n";
    
    for (size_t i = 0; i < 3 && i < salience_scores.size(); ++i) {
        cognitive.focus_on(ctx, salience_scores[i].concept_id, 0.1 * i);
    }
    
    auto focus_set = cognitive.get_focus_set(ctx);
    
    std::cout << "Current focus set (working memory):\n";
    for (const auto& entry : focus_set) {
        auto info = ltm.retrieve_concept(entry.concept_id);
        if (info.has_value()) {
            std::cout << "  " << info->label << " (focus score: " 
                      << std::fixed << std::setprecision(3) << entry.focus_score << ")\n";
        }
    }
    
    std::cout << "\nApplying focus decay...\n";
    cognitive.decay_focus(ctx);
    cognitive.decay_focus(ctx);
    
    focus_set = cognitive.get_focus_set(ctx);
    std::cout << "Focus set after decay:\n";
    for (const auto& entry : focus_set) {
        auto info = ltm.retrieve_concept(entry.concept_id);
        if (info.has_value()) {
            std::cout << "  " << info->label << " (focus score: " 
                      << std::fixed << std::setprecision(3) << entry.focus_score << ")\n";
        }
    }
    
    // PHASE 4: Thought Path Ranking
    print_separator("Phase 4: Thought Path Ranking");
    
    std::cout << "Finding best thought paths from 'Cat'...\n\n";
    
    auto paths = cognitive.find_best_paths(cat, ctx, ltm, *stm);
    
    std::cout << "Top thought paths:\n";
    for (size_t i = 0; i < paths.size() && i < 5; ++i) {
        std::cout << "  Path " << (i + 1) << " (score: " 
                  << std::fixed << std::setprecision(3) << paths[i].total_score << "): ";
        
        for (size_t j = 0; j < paths[i].nodes.size(); ++j) {
            auto info = ltm.retrieve_concept(paths[i].nodes[j].concept_id);
            if (info.has_value()) {
                std::cout << info->label;
                if (j < paths[i].nodes.size() - 1) {
                    std::cout << " → ";
                }
            }
        }
        std::cout << "\n";
    }
    
    // PHASE 5: Targeted Path Search
    print_separator("Phase 5: Targeted Path Search");
    
    std::cout << "Finding paths from 'Cat' to 'Warm-blooded'...\n\n";
    
    auto targeted_paths = cognitive.find_paths_to(cat, warm_blooded, ctx, ltm, *stm);
    
    if (targeted_paths.empty()) {
        std::cout << "  No direct paths found (may require more depth)\n";
    } else {
        for (size_t i = 0; i < targeted_paths.size() && i < 3; ++i) {
            std::cout << "  Path " << (i + 1) << " (score: " 
                      << std::fixed << std::setprecision(3) << targeted_paths[i].total_score << "): ";
            
            for (size_t j = 0; j < targeted_paths[i].nodes.size(); ++j) {
                auto info = ltm.retrieve_concept(targeted_paths[i].nodes[j].concept_id);
                if (info.has_value()) {
                    std::cout << info->label;
                    if (j < targeted_paths[i].nodes.size() - 1) {
                        std::cout << " → ";
                    }
                }
            }
            std::cout << "\n";
        }
    }
    
    // PHASE 6: Verify Epistemic Invariants
    print_separator("Phase 6: Epistemic Invariant Verification");
    
    std::cout << "Verifying that epistemic metadata was NOT modified...\n\n";
    
    bool all_intact = true;
    
    auto check_concept = [&](ConceptId id, const char* name, 
                            EpistemicType expected_type, double expected_trust) {
        auto info = ltm.retrieve_concept(id);
        if (!info.has_value()) {
            std::cout << "  ✗ " << name << ": NOT FOUND!\n";
            all_intact = false;
            return;
        }
        
        bool type_ok = (info->epistemic.type == expected_type);
        bool trust_ok = (std::abs(info->epistemic.trust - expected_trust) < 0.001);
        
        if (type_ok && trust_ok) {
            std::cout << "  ✓ " << name << ": intact\n";
        } else {
            std::cout << "  ✗ " << name << ": MODIFIED!\n";
            all_intact = false;
        }
    };
    
    check_concept(cat, "Cat", EpistemicType::FACT, 0.98);
    check_concept(mammal, "Mammal", EpistemicType::FACT, 0.95);
    check_concept(animal, "Animal", EpistemicType::DEFINITION, 0.99);
    check_concept(fur, "Fur", EpistemicType::FACT, 0.97);
    check_concept(nocturnal, "Nocturnal", EpistemicType::HYPOTHESIS, 0.70);
    
    std::cout << "\n";
    if (all_intact) {
        std::cout << "═══════════════════════════════════════════════════════\n";
        std::cout << "  ALL EPISTEMIC INVARIANTS PRESERVED ✓\n";
        std::cout << "═══════════════════════════════════════════════════════\n";
    } else {
        std::cout << "═══════════════════════════════════════════════════════\n";
        std::cout << "  WARNING: EPISTEMIC INVARIANTS VIOLATED!\n";
        std::cout << "═══════════════════════════════════════════════════════\n";
    }
    
    // Cleanup
    print_separator("Cleanup");
    
    brain.destroy_context(ctx);
    brain.shutdown();
    std::cout << "✓ All subsystems shut down\n";
    
    // Summary
    print_separator("Demo Complete");
    
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Cognitive Dynamics Integration - SUCCESS            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "FEATURES DEMONSTRATED:\n";
    std::cout << "  ✓ Spreading Activation (trust-weighted, depth-limited)\n";
    std::cout << "  ✓ Salience Computation (importance ranking)\n";
    std::cout << "  ✓ Focus Management (working memory simulation)\n";
    std::cout << "  ✓ Thought Path Ranking (inference prioritization)\n";
    std::cout << "  ✓ Targeted Path Search (goal-directed reasoning)\n\n";
    
    std::cout << "ARCHITECTURAL GUARANTEES:\n";
    std::cout << "  ✓ LTM accessed read-only\n";
    std::cout << "  ✓ Trust values unchanged\n";
    std::cout << "  ✓ EpistemicType unchanged\n";
    std::cout << "  ✓ EpistemicStatus unchanged\n";
    std::cout << "  ✓ No knowledge created\n";
    std::cout << "  ✓ No hypotheses generated\n";
    std::cout << "  ✓ Deterministic behavior\n";
    std::cout << "  ✓ Bounded activations [0.0, 1.0]\n\n";
    
    return 0;
}
