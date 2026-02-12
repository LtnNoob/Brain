#include "thinking_pipeline.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>

namespace brain19 {

ThinkingPipeline::ThinkingPipeline()
    : config_()
{}

ThinkingPipeline::ThinkingPipeline(Config config)
    : config_(std::move(config))
{}

ThinkingResult ThinkingPipeline::execute(
    const std::vector<ConceptId>& seed_concepts,
    ContextId context,
    LongTermMemory& ltm,
    ShortTermMemory& stm,
    BrainController& brain,
    CognitiveDynamics& cognitive,
    CuriosityEngine& curiosity,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    UnderstandingLayer* understanding,
    KanValidator* kan_validator,
    GlobalDynamicsOperator* gdo)
{
    auto start = std::chrono::steady_clock::now();
    ThinkingResult result;

    if (seed_concepts.empty()) {
        return result;
    }

    // Step 1: Activate seed concepts in STM
    step_activate_seeds(seed_concepts, context, stm, brain);
    result.steps_completed = 1;

    // Step 2: Spreading Activation
    step_spreading(seed_concepts, context, cognitive, ltm, stm);
    result.steps_completed = 2;

    // Step 2.5: FocusCursor traversal (optional)
    if (config_.enable_focus_cursor) {
        // Augment seeds with GDO activation landscape
        std::vector<ConceptId> cursor_seeds = seed_concepts;
        if (gdo) {
            auto gdo_top = gdo->get_activation_snapshot(3);
            for (const auto& [cid, act] : gdo_top) {
                bool already = false;
                for (ConceptId s : cursor_seeds) {
                    if (s == cid) { already = true; break; }
                }
                if (!already) cursor_seeds.push_back(cid);
            }
        }

        GoalState default_goal = GoalState::exploration_goal({}, "");
        auto qr = step_focus_cursor(
            cursor_seeds, context, ltm, stm, registry, embeddings, default_goal);
        if (!qr.best_chain.empty()) {
            result.cursor_result = qr.best_chain;
            // Feed cursor result back to GDO
            if (gdo) gdo->feed_traversal_result(qr.best_chain);
        }
    }

    // Gather active concepts
    result.activated_concepts = stm.get_active_concepts(context, 0.05);

    // Step 3: Compute Salience + Focus
    result.top_salient = step_salience(result.activated_concepts, context, cognitive, ltm, stm);
    cognitive.init_focus(context);
    for (auto& s : result.top_salient) {
        cognitive.focus_on(context, s.concept_id, s.salience);
    }
    result.steps_completed = 3;

    // Step 4-5: Generate and combine RelevanceMaps
    result.combined_relevance = step_relevance(result.top_salient, registry, embeddings, ltm);
    result.steps_completed = 5;

    // Step 6: Find ThoughtPaths
    result.best_paths = step_thought_paths(seed_concepts, context, cognitive, ltm, stm);
    result.steps_completed = 6;

    // Step 7: CuriosityEngine
    if (config_.enable_curiosity) {
        result.curiosity_triggers = step_curiosity(context, curiosity, stm);
        result.generated_goals = GoalGenerator::from_triggers(result.curiosity_triggers);
    }
    result.steps_completed = 7;

    // Step 8: UnderstandingLayer
    if (config_.enable_understanding && understanding) {
        std::vector<ConceptId> salient_ids;
        salient_ids.reserve(result.top_salient.size());
        for (auto& s : result.top_salient) {
            salient_ids.push_back(s.concept_id);
        }
        result.understanding = step_understanding(
            salient_ids, context, *understanding, cognitive, ltm, stm);
    }
    result.steps_completed = 8;

    // Step 9: KAN-LLM Validation
    if (config_.enable_kan_validation && kan_validator &&
        !result.understanding.hypothesis_proposals.empty()) {
        result.validated_hypotheses = step_kan_validation(
            result.understanding.hypothesis_proposals, *kan_validator);
    }
    result.steps_completed = 9;

    // Step 10: Complete
    result.steps_completed = 10;
    auto end = std::chrono::steady_clock::now();
    result.total_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// ─── Step Implementations ────────────────────────────────────────────────────

void ThinkingPipeline::step_activate_seeds(
    const std::vector<ConceptId>& seeds, ContextId ctx,
    ShortTermMemory& /*stm*/, BrainController& brain)
{
    for (auto cid : seeds) {
        brain.activate_concept_in_context(
            ctx, cid, config_.initial_activation, ActivationClass::CORE_KNOWLEDGE);
    }
}

SpreadingStats ThinkingPipeline::step_spreading(
    const std::vector<ConceptId>& seeds, ContextId ctx,
    CognitiveDynamics& cognitive, LongTermMemory& ltm, ShortTermMemory& stm)
{
    return cognitive.spread_activation_multi(
        seeds, config_.initial_activation, ctx, ltm, stm);
}

std::vector<SalienceScore> ThinkingPipeline::step_salience(
    const std::vector<ConceptId>& active, ContextId ctx,
    CognitiveDynamics& cognitive, LongTermMemory& ltm, ShortTermMemory& stm)
{
    return cognitive.get_top_k_salient(
        active, config_.top_k_salient, ctx, ltm, stm);
}

RelevanceMap ThinkingPipeline::step_relevance(
    const std::vector<SalienceScore>& salient,
    MicroModelRegistry& registry, EmbeddingManager& embeddings,
    LongTermMemory& ltm)
{
    std::vector<RelevanceMap> maps;
    size_t count = std::min(salient.size(), config_.max_relevance_maps);

    for (size_t i = 0; i < count; ++i) {
        auto cid = salient[i].concept_id;
        if (registry.has_model(cid)) {
            auto map = RelevanceMap::compute(
                cid, registry, embeddings, ltm,
                RelationType::IS_A, "query");
            maps.push_back(std::move(map));
        }
    }

    if (maps.empty()) {
        return RelevanceMap{};
    }
    if (maps.size() == 1) {
        return std::move(maps[0]);
    }

    // Combine with overlay for creativity
    return RelevanceMap::combine(maps, OverlayMode::WEIGHTED_AVERAGE);
}

std::vector<ThoughtPath> ThinkingPipeline::step_thought_paths(
    const std::vector<ConceptId>& seeds, ContextId ctx,
    CognitiveDynamics& cognitive, LongTermMemory& ltm, ShortTermMemory& stm)
{
    std::vector<ThoughtPath> all_paths;
    for (auto cid : seeds) {
        auto paths = cognitive.find_best_paths(cid, ctx, ltm, stm);
        for (auto& p : paths) {
            all_paths.push_back(std::move(p));
        }
    }
    // Sort by score, keep top
    std::sort(all_paths.begin(), all_paths.end());
    if (all_paths.size() > 20) {
        all_paths.resize(20);
    }
    return all_paths;
}

std::vector<CuriosityTrigger> ThinkingPipeline::step_curiosity(
    ContextId ctx, CuriosityEngine& curiosity, ShortTermMemory& stm)
{
    SystemObservation obs;
    obs.context_id = ctx;
    obs.active_concept_count = stm.debug_active_concept_count(ctx);
    obs.active_relation_count = stm.debug_active_relation_count(ctx);
    return curiosity.observe_and_generate_triggers({obs});
}

UnderstandingLayer::UnderstandingResult ThinkingPipeline::step_understanding(
    const std::vector<ConceptId>& salient_ids, ContextId ctx,
    UnderstandingLayer& understanding, CognitiveDynamics& cognitive,
    LongTermMemory& ltm, ShortTermMemory& stm)
{
    if (salient_ids.empty()) {
        return {};
    }
    return understanding.perform_understanding_cycle(
        salient_ids[0], cognitive, ltm, stm, ctx);
}

std::vector<ValidationResult> ThinkingPipeline::step_kan_validation(
    const std::vector<HypothesisProposal>& hypotheses,
    KanValidator& validator)
{
    std::vector<ValidationResult> results;
    for (auto& hyp : hypotheses) {
        try {
            results.push_back(validator.validate(hyp));
        } catch (const std::exception& e) {
            std::cerr << "[ThinkingPipeline] KAN validation failed: " << e.what() << "\n";
        }
    }
    return results;
}

// ─── FocusCursor Step ─────────────────────────────────────────────────────────

QueryResult ThinkingPipeline::step_focus_cursor(
    const std::vector<ConceptId>& seeds,
    ContextId ctx,
    LongTermMemory& ltm,
    ShortTermMemory& stm,
    MicroModelRegistry& registry,
    EmbeddingManager& embeddings,
    const GoalState& goal)
{
    FocusCursorManager mgr(ltm, registry, embeddings, stm, config_.cursor_config);

    Vec10 query_context{};
    auto qr = mgr.process_seeds(seeds, query_context, goal);

    // Persist best chain to STM
    if (!qr.best_chain.empty()) {
        mgr.persist_to_stm(ctx, qr.best_chain);
    }

    return qr;
}

// ─── Execute with Goal ───────────────────────────────────────────────────────

ThinkingResult ThinkingPipeline::execute_with_goal(
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
    KanValidator* kan_validator,
    GlobalDynamicsOperator* gdo)
{
    auto start = std::chrono::steady_clock::now();
    ThinkingResult result;

    if (seed_concepts.empty()) {
        return result;
    }

    // Step 1: Activate seed concepts in STM
    step_activate_seeds(seed_concepts, context, stm, brain);
    result.steps_completed = 1;

    // Step 2: Spreading Activation
    step_spreading(seed_concepts, context, cognitive, ltm, stm);
    result.steps_completed = 2;

    // Step 2.5: FocusCursor with explicit goal
    if (config_.enable_focus_cursor) {
        // Augment seeds with GDO landscape
        std::vector<ConceptId> cursor_seeds = seed_concepts;
        if (gdo) {
            auto gdo_top = gdo->get_activation_snapshot(3);
            for (const auto& [cid, act] : gdo_top) {
                bool already = false;
                for (ConceptId s : cursor_seeds) {
                    if (s == cid) { already = true; break; }
                }
                if (!already) cursor_seeds.push_back(cid);
            }
        }

        auto qr = step_focus_cursor(
            cursor_seeds, context, ltm, stm, registry, embeddings, goal);
        if (!qr.best_chain.empty()) {
            result.cursor_result = qr.best_chain;
            if (gdo) gdo->feed_traversal_result(qr.best_chain);
        }
        result.final_goal_state = goal;
    }

    // Gather active concepts
    result.activated_concepts = stm.get_active_concepts(context, 0.05);

    // Step 3: Compute Salience + Focus
    result.top_salient = step_salience(result.activated_concepts, context, cognitive, ltm, stm);
    cognitive.init_focus(context);
    for (auto& s : result.top_salient) {
        cognitive.focus_on(context, s.concept_id, s.salience);
    }
    result.steps_completed = 3;

    // Step 4-5: Generate and combine RelevanceMaps
    result.combined_relevance = step_relevance(result.top_salient, registry, embeddings, ltm);
    result.steps_completed = 5;

    // Step 6: Find ThoughtPaths
    result.best_paths = step_thought_paths(seed_concepts, context, cognitive, ltm, stm);
    result.steps_completed = 6;

    // Step 7: CuriosityEngine
    if (config_.enable_curiosity) {
        result.curiosity_triggers = step_curiosity(context, curiosity, stm);
        result.generated_goals = GoalGenerator::from_triggers(result.curiosity_triggers);
    }
    result.steps_completed = 7;

    // Step 8: UnderstandingLayer
    if (config_.enable_understanding && understanding) {
        std::vector<ConceptId> salient_ids;
        salient_ids.reserve(result.top_salient.size());
        for (auto& s : result.top_salient) {
            salient_ids.push_back(s.concept_id);
        }
        result.understanding = step_understanding(
            salient_ids, context, *understanding, cognitive, ltm, stm);
    }
    result.steps_completed = 8;

    // Step 9: KAN-LLM Validation
    if (config_.enable_kan_validation && kan_validator &&
        !result.understanding.hypothesis_proposals.empty()) {
        result.validated_hypotheses = step_kan_validation(
            result.understanding.hypothesis_proposals, *kan_validator);
    }
    result.steps_completed = 9;

    // Step 10: Complete
    result.steps_completed = 10;
    auto end = std::chrono::steady_clock::now();
    result.total_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

} // namespace brain19
