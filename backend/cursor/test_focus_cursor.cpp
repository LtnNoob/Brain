// Unit Test: FocusCursor + FocusCursorManager + GoalState
// Uses a hardcoded mini-graph: Eis →CAUSES→ Schmelzen →CAUSES→ Wasser →HAS_PROPERTY→ Fluessig
//
// Build:
//   make test_focus_cursor

#include "focus_cursor.hpp"
#include "focus_cursor_manager.hpp"
#include "goal_state.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../cmodel/concept_trainer.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../memory/stm.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace brain19;

// =============================================================================
// Helper: Build a mini knowledge graph
// =============================================================================
struct MiniGraph {
    LongTermMemory ltm;
    ConceptModelRegistry registry;
    EmbeddingManager embeddings;
    ConceptTrainer trainer;
    ShortTermMemory stm;

    ConceptId eis, schmelzen, wasser, fluessig, dampf;

    MiniGraph() {
        EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);

        eis       = ltm.store_concept("Eis", "Gefrorenes Wasser", meta);
        schmelzen = ltm.store_concept("Schmelzen", "Phasenuebergang fest zu fluessig", meta);
        wasser    = ltm.store_concept("Wasser", "H2O in fluessiger Form", meta);
        fluessig  = ltm.store_concept("Fluessig", "Aggregatzustand fluessig", meta);
        dampf     = ltm.store_concept("Dampf", "H2O in gasfoermiger Form", meta);

        // Chain: Eis →CAUSES→ Schmelzen →CAUSES→ Wasser →HAS_PROPERTY→ Fluessig
        ltm.add_relation(eis, schmelzen, RelationType::CAUSES, 0.9);
        ltm.add_relation(schmelzen, wasser, RelationType::CAUSES, 0.85);
        ltm.add_relation(wasser, fluessig, RelationType::HAS_PROPERTY, 0.8);
        ltm.add_relation(wasser, dampf, RelationType::CAUSES, 0.7);  // Branch

        // Ensure MicroModels exist for all concepts
        registry.ensure_models_for(ltm);

        // Train MicroModels on graph structure
        trainer.train_all(registry, embeddings, ltm);
    }
};

// =============================================================================
// Test 1: GoalState basics
// =============================================================================
void test_goal_state() {
    std::cout << "TEST: GoalState basics... ";

    GoalState gs;
    assert(!gs.is_complete());
    assert(gs.goal_type == GoalType::EXPLORATION);

    // After enough steps, exploration goal completes
    gs.update_progress({1, 2, 3, 4, 5}, 10);
    assert(gs.completion_metric > 0.5);

    // Definition goal: complete when target found
    Vec10 emb{};
    auto dg = GoalState::definition_goal(42, emb, "Was ist X?");
    assert(dg.goal_type == GoalType::DEFINITION);
    assert(!dg.is_complete());

    dg.update_progress({1, 2, 42}, 3);  // Target 42 found
    assert(dg.completion_metric >= 1.0);
    assert(dg.is_complete());

    std::cout << "PASS\n";
}

// =============================================================================
// Test 2: FocusCursor seed and position
// =============================================================================
void test_cursor_seed(MiniGraph& g) {
    std::cout << "TEST: FocusCursor seed... ";

    FocusCursor cursor(g.ltm, g.registry, g.embeddings);
    assert(!cursor.is_seeded());

    cursor.seed(g.eis);
    assert(cursor.is_seeded());
    assert(cursor.position() == g.eis);
    assert(cursor.depth() == 0);
    assert(!cursor.is_terminated());

    auto view = cursor.get_view();
    assert(view.current == g.eis);
    assert(view.depth == 0);

    std::cout << "PASS\n";
}

// =============================================================================
// Test 3: FocusCursor step traverses to next concept
// =============================================================================
void test_cursor_step(MiniGraph& g) {
    std::cout << "TEST: FocusCursor step... ";

    FocusCursor cursor(g.ltm, g.registry, g.embeddings);
    cursor.seed(g.eis);

    auto next = cursor.step();
    assert(next.has_value());
    // Should move to Schmelzen (only outgoing CAUSES with weight 0.9)
    assert(*next == g.schmelzen);
    assert(cursor.depth() == 1);

    std::cout << "PASS (stepped to concept " << *next << ")\n";
}

// =============================================================================
// Test 4: FocusCursor deepen runs full chain
// =============================================================================
void test_cursor_deepen(MiniGraph& g) {
    std::cout << "TEST: FocusCursor deepen... ";

    FocusCursor cursor(g.ltm, g.registry, g.embeddings);
    cursor.seed(g.eis);

    auto result = cursor.deepen();
    assert(!result.empty());
    assert(result.concept_sequence.size() >= 2);  // At least Eis + one more
    assert(result.concept_sequence[0] == g.eis);
    assert(result.chain_score > 0.0);

    std::cout << "PASS (chain length=" << result.concept_sequence.size()
              << ", score=" << result.chain_score << ")\n";

    // Print the chain
    std::cout << "  Chain: ";
    for (size_t i = 0; i < result.concept_sequence.size(); ++i) {
        auto c = g.ltm.retrieve_concept(result.concept_sequence[i]);
        if (c) std::cout << c->label;
        if (i + 1 < result.concept_sequence.size()) {
            std::cout << " -> ";
        }
    }
    std::cout << "\n";
}

// =============================================================================
// Test 5: FocusCursor backtrack
// =============================================================================
void test_cursor_backtrack(MiniGraph& g) {
    std::cout << "TEST: FocusCursor backtrack... ";

    FocusCursor cursor(g.ltm, g.registry, g.embeddings);
    cursor.seed(g.eis);
    cursor.step();  // Move to Schmelzen

    assert(cursor.position() == g.schmelzen);

    bool ok = cursor.backtrack();
    assert(ok);
    assert(cursor.position() == g.eis);
    assert(cursor.depth() == 0);

    // Can't backtrack past seed
    ok = cursor.backtrack();
    assert(!ok);

    std::cout << "PASS\n";
}

// =============================================================================
// Test 6: FocusCursor termination at max_depth
// =============================================================================
void test_cursor_max_depth(MiniGraph& g) {
    std::cout << "TEST: FocusCursor max_depth termination... ";

    FocusCursorConfig config;
    config.max_depth = 2;  // Limit to 2 steps

    FocusCursor cursor(g.ltm, g.registry, g.embeddings, config);
    cursor.seed(g.eis);

    auto result = cursor.deepen();
    // Chain: Eis (seed, depth=0) + 2 more = max 3 concepts
    assert(result.concept_sequence.size() <= 3);

    std::cout << "PASS (chain length=" << result.concept_sequence.size() << ")\n";
}

// =============================================================================
// Test 7: FocusCursorManager process_seeds
// =============================================================================
void test_manager_process_seeds(MiniGraph& g) {
    std::cout << "TEST: FocusCursorManager process_seeds... ";

    FocusCursorManager mgr(g.ltm, g.registry, g.embeddings, g.stm);

    Vec10 ctx{};
    auto result = mgr.process_seeds({g.eis}, ctx);

    assert(!result.chains.empty());
    assert(!result.best_chain.empty());
    assert(!result.all_activated.empty());

    std::cout << "PASS (chains=" << result.chains.size()
              << ", best_len=" << result.best_chain.concept_sequence.size()
              << ", activated=" << result.all_activated.size() << ")\n";
}

// =============================================================================
// Test 8: FocusCursorManager persist_to_stm
// =============================================================================
void test_manager_persist_stm(MiniGraph& g) {
    std::cout << "TEST: FocusCursorManager persist_to_stm... ";

    FocusCursorManager mgr(g.ltm, g.registry, g.embeddings, g.stm);

    ContextId ctx = g.stm.create_context();
    Vec10 emb{};
    auto result = mgr.process_seeds({g.eis}, emb);

    mgr.persist_to_stm(ctx, result.best_chain);

    // Check that seed concept is activated in STM
    double activation = g.stm.get_concept_activation(ctx, g.eis);
    assert(activation > 0.0);

    std::cout << "PASS (Eis activation=" << activation << ")\n";
}

// =============================================================================
// Test 9: Cycle detection — cursor should not revisit concepts
// =============================================================================
void test_cursor_no_cycles(MiniGraph& g) {
    std::cout << "TEST: FocusCursor cycle detection... ";

    // Add a back-edge: Wasser →CAUSES→ Eis (creates cycle)
    g.ltm.add_relation(g.wasser, g.eis, RelationType::CAUSES, 0.9);

    FocusCursor cursor(g.ltm, g.registry, g.embeddings);
    cursor.seed(g.eis);
    auto result = cursor.deepen();

    // Check no concept appears twice
    std::set<ConceptId> seen;
    for (ConceptId cid : result.concept_sequence) {
        assert(seen.find(cid) == seen.end() && "Cycle detected: concept revisited!");
        seen.insert(cid);
    }

    std::cout << "PASS (no cycles in chain of " << result.concept_sequence.size() << ")\n";
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n=== FocusCursor Unit Tests ===\n\n";

    test_goal_state();

    MiniGraph g;
    test_cursor_seed(g);
    test_cursor_step(g);
    test_cursor_deepen(g);
    test_cursor_backtrack(g);
    test_cursor_max_depth(g);
    test_manager_process_seeds(g);
    test_manager_persist_stm(g);
    test_cursor_no_cycles(g);

    std::cout << "\n=== ALL TESTS PASSED ===\n\n";
    return 0;
}
