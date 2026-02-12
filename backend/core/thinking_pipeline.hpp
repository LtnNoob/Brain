#pragma once

#include "../common/types.hpp"
#include "../cognitive/cognitive_dynamics.hpp"
#include "../curiosity/curiosity_engine.hpp"
#include "../curiosity/goal_generator.hpp"
#include "../micromodel/relevance_map.hpp"
#include "../micromodel/micro_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../understanding/understanding_layer.hpp"
#include "../hybrid/kan_validator.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/stm.hpp"
#include "../memory/brain_controller.hpp"
#include "../cursor/traversal_types.hpp"
#include "../cursor/goal_state.hpp"
#include "../cursor/focus_cursor_manager.hpp"

#include <optional>
#include <vector>
#include <string>

namespace brain19 {

// =============================================================================
// THINKING RESULT
// =============================================================================

struct ThinkingResult {
    std::vector<ConceptId> activated_concepts;
    std::vector<SalienceScore> top_salient;
    std::vector<ThoughtPath> best_paths;
    std::vector<CuriosityTrigger> curiosity_triggers;
    RelevanceMap combined_relevance;
    UnderstandingLayer::UnderstandingResult understanding;
    std::vector<ValidationResult> validated_hypotheses;

    // Pipeline statistics
    size_t steps_completed = 0;
    double total_duration_ms = 0.0;

    // --- FocusCursor results ---
    std::optional<TraversalResult> cursor_result;
    std::optional<GoalState> final_goal_state;

    // --- Generated goals from curiosity ---
    std::vector<GoalState> generated_goals;
};

// =============================================================================
// THINKING PIPELINE
// =============================================================================
//
// Orchestrates a complete "thinking cycle" — the heart of Brain19.
//
// Pipeline Steps:
// 1. Activate seed concepts in STM
// 2. Spreading Activation (CognitiveDynamics)
// 3. Compute Salience + Focus
// 4. Generate RelevanceMaps (MicroModels)
// 5. Combine RelevanceMaps (Overlay for creativity)
// 6. Find ThoughtPaths
// 7. Run CuriosityEngine
// 8. Run UnderstandingLayer (MiniLLMs)
// 9. KAN-LLM Validation (Phase 7 Hybrid)
// 10. Return complete result
//
class ThinkingPipeline {
public:
    struct Config {
        double initial_activation = 0.8;
        size_t top_k_salient = 10;
        size_t max_relevance_maps = 5;
        bool enable_understanding = true;
        bool enable_kan_validation = true;
        bool enable_curiosity = true;

        // FocusCursor integration
        bool enable_focus_cursor = true;
        FocusCursorConfig cursor_config{};
    };

    ThinkingPipeline();
    explicit ThinkingPipeline(Config config);
    ~ThinkingPipeline() = default;

    // Execute a full thinking cycle
    ThinkingResult execute(
        const std::vector<ConceptId>& seed_concepts,
        ContextId context,
        LongTermMemory& ltm,
        ShortTermMemory& stm,
        BrainController& brain,
        CognitiveDynamics& cognitive,
        CuriosityEngine& curiosity,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        UnderstandingLayer* understanding,  // nullable if no LLM
        KanValidator* kan_validator          // nullable if no KAN validation
    );

    // Execute with explicit goal for goal-directed traversal
    ThinkingResult execute_with_goal(
        const std::vector<ConceptId>& seed_concepts,
        GoalState goal,
        ContextId context,
        LongTermMemory& ltm,
        ShortTermMemory& stm,
        BrainController& brain,
        CognitiveDynamics& cognitive,
        CuriosityEngine& curiosity,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        UnderstandingLayer* understanding,
        KanValidator* kan_validator
    );

    const Config& get_config() const { return config_; }

private:
    Config config_;

    // Pipeline step implementations
    void step_activate_seeds(
        const std::vector<ConceptId>& seeds, ContextId ctx,
        ShortTermMemory& stm, BrainController& brain);

    SpreadingStats step_spreading(
        const std::vector<ConceptId>& seeds, ContextId ctx,
        CognitiveDynamics& cognitive, LongTermMemory& ltm, ShortTermMemory& stm);

    std::vector<SalienceScore> step_salience(
        const std::vector<ConceptId>& active, ContextId ctx,
        CognitiveDynamics& cognitive, LongTermMemory& ltm, ShortTermMemory& stm);

    RelevanceMap step_relevance(
        const std::vector<SalienceScore>& salient,
        MicroModelRegistry& registry, EmbeddingManager& embeddings,
        LongTermMemory& ltm);

    std::vector<ThoughtPath> step_thought_paths(
        const std::vector<ConceptId>& seeds, ContextId ctx,
        CognitiveDynamics& cognitive, LongTermMemory& ltm, ShortTermMemory& stm);

    std::vector<CuriosityTrigger> step_curiosity(
        ContextId ctx, CuriosityEngine& curiosity, ShortTermMemory& stm);

    UnderstandingLayer::UnderstandingResult step_understanding(
        const std::vector<ConceptId>& salient_ids, ContextId ctx,
        UnderstandingLayer& understanding, CognitiveDynamics& cognitive,
        LongTermMemory& ltm, ShortTermMemory& stm);

    std::vector<ValidationResult> step_kan_validation(
        const std::vector<HypothesisProposal>& hypotheses,
        KanValidator& validator);

    // Step 2.5: FocusCursor traversal (after spreading, before salience)
    QueryResult step_focus_cursor(
        const std::vector<ConceptId>& seeds,
        ContextId ctx,
        LongTermMemory& ltm,
        ShortTermMemory& stm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        const GoalState& goal);
};

} // namespace brain19
