// Unit Test: ThinkingPipeline with FocusCursor integration
//
// Verifies that the FocusCursor step (2.5) runs within the pipeline
// and produces cursor_result in ThinkingResult.
//
// Build:
//   make test_pipeline_cursor

#include "thinking_pipeline.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/stm.hpp"
#include "../memory/brain_controller.hpp"
#include "../cognitive/cognitive_dynamics.hpp"
#include "../curiosity/curiosity_engine.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../micromodel/micro_trainer.hpp"
#include "../understanding/understanding_layer.hpp"
#include "../cursor/template_engine.hpp"
#include <cassert>
#include <iostream>

using namespace brain19;

// =============================================================================
// Helper: Build a mini knowledge graph with all subsystems
// =============================================================================
struct TestEnv {
    LongTermMemory ltm;
    BrainController brain;
    CognitiveDynamics cognitive;
    CuriosityEngine curiosity;
    MicroModelRegistry registry;
    EmbeddingManager embeddings;
    MicroTrainer trainer;
    UnderstandingLayer understanding;

    ConceptId eis, schmelzen, wasser, fluessig;
    ContextId ctx;

    TestEnv() {
        brain.initialize();

        EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);

        eis       = ltm.store_concept("Eis", "Gefrorenes Wasser", meta);
        schmelzen = ltm.store_concept("Schmelzen", "Phasenuebergang", meta);
        wasser    = ltm.store_concept("Wasser", "H2O fluessig", meta);
        fluessig  = ltm.store_concept("Fluessig", "Aggregatzustand", meta);

        ltm.add_relation(eis, schmelzen, RelationType::CAUSES, 0.9);
        ltm.add_relation(schmelzen, wasser, RelationType::CAUSES, 0.85);
        ltm.add_relation(wasser, fluessig, RelationType::HAS_PROPERTY, 0.8);

        registry.ensure_models_for(ltm);
        trainer.train_all(registry, embeddings, ltm);

        // Register stub MiniLLM
        understanding.register_mini_llm(std::make_unique<StubMiniLLM>());

        ctx = brain.create_context();
    }

    ~TestEnv() {
        brain.destroy_context(ctx);
        brain.shutdown();
    }
};

// =============================================================================
// Test 1: Pipeline with FocusCursor enabled produces cursor_result
// =============================================================================
void test_pipeline_cursor_enabled() {
    std::cout << "TEST: Pipeline with FocusCursor enabled... ";

    TestEnv env;

    ThinkingPipeline::Config cfg;
    cfg.enable_focus_cursor = true;
    cfg.enable_understanding = false;  // Skip for speed
    cfg.enable_kan_validation = false;

    ThinkingPipeline pipeline(cfg);

    auto result = pipeline.execute(
        {env.eis}, env.ctx,
        env.ltm, *env.brain.get_stm_mutable(), env.brain,
        env.cognitive, env.curiosity,
        env.registry, env.embeddings,
        &env.understanding, nullptr
    );

    assert(result.steps_completed == 10);
    assert(result.cursor_result.has_value());
    assert(!result.cursor_result->concept_sequence.empty());
    assert(result.cursor_result->concept_sequence[0] == env.eis);
    assert(result.cursor_result->chain_score > 0.0);

    std::cout << "PASS (chain length=" << result.cursor_result->concept_sequence.size()
              << ", score=" << result.cursor_result->chain_score << ")\n";

    // Print the chain
    std::cout << "  Pipeline chain: ";
    for (size_t i = 0; i < result.cursor_result->concept_sequence.size(); ++i) {
        auto c = env.ltm.retrieve_concept(result.cursor_result->concept_sequence[i]);
        if (c) std::cout << c->label;
        if (i + 1 < result.cursor_result->concept_sequence.size()) std::cout << " -> ";
    }
    std::cout << "\n";
}

// =============================================================================
// Test 2: Pipeline with FocusCursor disabled — no cursor_result
// =============================================================================
void test_pipeline_cursor_disabled() {
    std::cout << "TEST: Pipeline with FocusCursor disabled... ";

    TestEnv env;

    ThinkingPipeline::Config cfg;
    cfg.enable_focus_cursor = false;
    cfg.enable_understanding = false;
    cfg.enable_kan_validation = false;

    ThinkingPipeline pipeline(cfg);

    auto result = pipeline.execute(
        {env.eis}, env.ctx,
        env.ltm, *env.brain.get_stm_mutable(), env.brain,
        env.cognitive, env.curiosity,
        env.registry, env.embeddings,
        &env.understanding, nullptr
    );

    assert(result.steps_completed == 10);
    assert(!result.cursor_result.has_value());

    std::cout << "PASS (no cursor_result)\n";
}

// =============================================================================
// Test 3: execute_with_goal provides goal state
// =============================================================================
void test_pipeline_with_goal() {
    std::cout << "TEST: Pipeline execute_with_goal... ";

    TestEnv env;

    ThinkingPipeline::Config cfg;
    cfg.enable_focus_cursor = true;
    cfg.enable_understanding = false;
    cfg.enable_kan_validation = false;

    ThinkingPipeline pipeline(cfg);

    Vec10 emb{};
    GoalState goal = GoalState::definition_goal(env.wasser, emb, "Was ist Wasser?");

    auto result = pipeline.execute_with_goal(
        {env.eis}, goal, env.ctx,
        env.ltm, *env.brain.get_stm_mutable(), env.brain,
        env.cognitive, env.curiosity,
        env.registry, env.embeddings,
        &env.understanding, nullptr
    );

    assert(result.steps_completed == 10);
    assert(result.cursor_result.has_value());
    assert(result.final_goal_state.has_value());
    assert(result.final_goal_state->goal_type == GoalType::DEFINITION);

    std::cout << "PASS (goal type=DEFINITION)\n";
}

// =============================================================================
// Test 4: Template Engine can generate from pipeline result
// =============================================================================
void test_pipeline_template_integration() {
    std::cout << "TEST: Pipeline + Template Engine... ";

    TestEnv env;

    ThinkingPipeline::Config cfg;
    cfg.enable_focus_cursor = true;
    cfg.enable_understanding = false;
    cfg.enable_kan_validation = false;

    ThinkingPipeline pipeline(cfg);

    auto result = pipeline.execute(
        {env.eis}, env.ctx,
        env.ltm, *env.brain.get_stm_mutable(), env.brain,
        env.cognitive, env.curiosity,
        env.registry, env.embeddings,
        &env.understanding, nullptr
    );

    assert(result.cursor_result.has_value());

    // Feed cursor result to Template Engine
    TemplateEngine te(env.ltm);
    auto tmpl = te.generate(*result.cursor_result);

    assert(tmpl.sentences_generated > 0);
    assert(!tmpl.text.empty());
    assert(tmpl.text.find("Eis") != std::string::npos);

    std::cout << "PASS\n  Template output: \"" << tmpl.text << "\"\n";
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n=== ThinkingPipeline + FocusCursor Integration Tests ===\n\n";

    test_pipeline_cursor_enabled();
    test_pipeline_cursor_disabled();
    test_pipeline_with_goal();
    test_pipeline_template_integration();

    std::cout << "\n=== ALL TESTS PASSED ===\n\n";
    return 0;
}
