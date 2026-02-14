#include "thinking_pipeline.hpp"
#include "../cmodel/concept_pattern_engine.hpp"
#include "../hybrid/kan_graph_monitor.hpp"
#include "../hybrid/topology_router.hpp"
#include "../hybrid/refinement_loop.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <unordered_set>

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
    ConceptModelRegistry& registry,
    EmbeddingManager& embeddings,
    UnderstandingLayer* understanding,
    KanValidator* kan_validator,
    GlobalDynamicsOperator* gdo,
    RefinementLoop* refinement_loop)
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

    // Build salient_ids for steps 7.5, 8, 8.5
    std::unordered_set<ConceptId> seen;
    std::vector<ConceptId> salient_ids;
    for (auto cid : seed_concepts) {
        if (seen.insert(cid).second) salient_ids.push_back(cid);
    }
    for (auto& s : result.top_salient) {
        if (seen.insert(s.concept_id).second) salient_ids.push_back(s.concept_id);
    }

    // Step 7.5: KAN Graph Scan (Topology A — detect anomalies)
    if (config_.enable_topology_a) {
        result.kan_anomalies = step_kan_graph_scan(
            salient_ids, registry, embeddings, ltm);
    }

    // Step 8: UnderstandingLayer — pass ThoughtPaths for multi-hop reasoning
    if (config_.enable_understanding && understanding) {
        result.understanding = step_understanding(
            salient_ids, context, *understanding, cognitive, ltm, stm,
            result.best_paths);
    }
    result.steps_completed = 8;

    // Step 8.5: Topology A Investigation (KAN anomalies → hypotheses)
    if (config_.enable_topology_a && understanding && !result.kan_anomalies.empty()) {
        result.topology_a_hypotheses = step_topology_a(
            result.kan_anomalies, *understanding, ltm, stm, context);
        for (const auto& h : result.topology_a_hypotheses) {
            result.understanding.hypothesis_proposals.push_back(h);
        }
    }

    // Step 9: KAN-LLM Validation — pass LTM for chain validation
    if (config_.enable_kan_validation && kan_validator &&
        !result.understanding.hypothesis_proposals.empty()) {
        result.validated_hypotheses = step_kan_validation(
            result.understanding.hypothesis_proposals, *kan_validator, &ltm);
    }
    result.steps_completed = 9;

    // Step 9C: Topology C Refinement
    if (config_.enable_topology_c && refinement_loop && understanding &&
        !result.validated_hypotheses.empty()) {
        step_topology_c(result, *refinement_loop, *understanding, ltm, stm, context);
    }

    // Step 9.5A: Trust ceiling enforcement (Topology B)
    for (auto& vr : result.validated_hypotheses) {
        double ceiling = vr.validated
            ? HypothesisProposal::KAN_VALIDATED_TRUST_CEILING
            : HypothesisProposal::LLM_ONLY_TRUST_CEILING;
        double capped = std::min(vr.assessment.metadata.trust, ceiling);
        EpistemicType type = vr.validated
            ? vr.assessment.metadata.type
            : EpistemicType::SPECULATION;
        vr.assessment.metadata = EpistemicMetadata(
            type, vr.assessment.metadata.status, capped);
    }

    // Step 9.5B: Feed validation back to ConceptPatternEngines
    if (understanding && !result.validated_hypotheses.empty()) {
        understanding->for_each_mini_llm([&](MiniLLM& llm) {
            auto* kan_aware = dynamic_cast<ConceptPatternEngine*>(&llm);
            if (kan_aware) {
                kan_aware->train_from_validation(result.validated_hypotheses);
            }
        });
    }

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
    ConceptModelRegistry& registry, EmbeddingManager& embeddings,
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
    UnderstandingLayer& understanding, CognitiveDynamics& /*cognitive*/,
    LongTermMemory& ltm, ShortTermMemory& stm,
    const std::vector<ThoughtPath>& thought_paths)
{
    if (salient_ids.empty()) {
        return {};
    }

    // FIX: Use ALL salient concepts directly instead of re-doing spreading
    // activation from a single seed via perform_understanding_cycle().
    // The ThinkingPipeline already spread activation from all seeds (step 2)
    // and computed salience (step 3). Reuse that work.
    UnderstandingLayer::UnderstandingResult result;

    result.meaning_proposals = understanding.analyze_meaning(
        salient_ids, ltm, stm, ctx);

    result.hypothesis_proposals = understanding.propose_hypotheses(
        salient_ids, ltm, stm, ctx, thought_paths);

    result.contradiction_proposals = understanding.check_contradictions(
        salient_ids, ltm, stm, ctx);

    // Analogies: split salient concepts into two sets
    if (salient_ids.size() >= 4) {
        auto mid = salient_ids.size() / 2;
        std::vector<ConceptId> set_a(salient_ids.begin(), salient_ids.begin() + mid);
        std::vector<ConceptId> set_b(salient_ids.begin() + mid, salient_ids.end());
        result.analogy_proposals = understanding.find_analogies(
            set_a, set_b, ltm, stm, ctx);
    }

    result.total_proposals_generated =
        result.meaning_proposals.size() +
        result.hypothesis_proposals.size() +
        result.analogy_proposals.size() +
        result.contradiction_proposals.size();

    return result;
}

std::vector<ValidationResult> ThinkingPipeline::step_kan_validation(
    const std::vector<HypothesisProposal>& hypotheses,
    KanValidator& validator,
    LongTermMemory* ltm)
{
    std::vector<ValidationResult> results;
    for (const auto& hyp : hypotheses) {
        try {
            // Check if this is a multi-hop chain hypothesis
            bool is_chain = false;
            if (ltm && hyp.evidence_concepts.size() >= 3) {
                for (const auto& pat : hyp.detected_patterns) {
                    if (pat == "multi-hop-chain") { is_chain = true; break; }
                }
            }

            if (is_chain) {
                auto chain_result = validator.validate_chain(hyp, *ltm);
                if (chain_result.chain_valid) {
                    // Build ValidationResult from chain validation
                    EpistemicAssessment chain_assess(
                        EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE,
                                          chain_result.geometric_mean_trust),
                        chain_result.geometric_mean_trust, true,
                        static_cast<size_t>(chain_result.edge_results.size()), 0.0,
                        chain_result.chain_summary, true
                    );
                    results.emplace_back(
                        true, std::move(chain_assess),
                        RelationshipPattern::LINEAR, nullptr,
                        "Chain[" + std::to_string(hyp.evidence_concepts.size()) + " nodes]: " + chain_result.chain_summary
                    );
                } else {
                    // Chain invalid, fallback to standard validation
                    results.push_back(validator.validate(hyp));
                }
            } else {
                results.push_back(validator.validate(hyp));
            }
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
    ConceptModelRegistry& registry,
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

// ─── Topology A+C Steps ──────────────────────────────────────────────────────

std::vector<InvestigationRequest> ThinkingPipeline::step_kan_graph_scan(
    const std::vector<ConceptId>& salient_ids,
    ConceptModelRegistry& registry,
    EmbeddingManager& embeddings,
    LongTermMemory& ltm)
{
    KanGraphMonitor monitor(registry, embeddings);
    return monitor.scan(salient_ids, ltm);
}

std::vector<HypothesisProposal> ThinkingPipeline::step_topology_a(
    const std::vector<InvestigationRequest>& anomalies,
    UnderstandingLayer& understanding,
    LongTermMemory& ltm,
    ShortTermMemory& stm,
    ContextId context)
{
    std::vector<HypothesisProposal> all_hypotheses;

    // Use ConceptPatternEngines to investigate anomalies
    understanding.for_each_mini_llm([&](MiniLLM& llm) {
        auto* kan_aware = dynamic_cast<ConceptPatternEngine*>(&llm);
        if (kan_aware) {
            auto hypotheses = kan_aware->investigate_anomalies(
                anomalies, ltm, stm, context);
            for (auto& h : hypotheses) {
                all_hypotheses.push_back(std::move(h));
            }
        }
    });

    return all_hypotheses;
}

void ThinkingPipeline::step_topology_c(
    ThinkingResult& result,
    RefinementLoop& refinement_loop,
    UnderstandingLayer& understanding,
    LongTermMemory& ltm,
    ShortTermMemory& stm,
    ContextId context)
{
    TopologyRouter router;
    auto decision = router.route(result.kan_anomalies, result.validated_hypotheses);

    if (decision.refine_indices.empty()) return;

    for (size_t idx : decision.refine_indices) {
        if (idx >= result.validated_hypotheses.size()) continue;
        if (idx >= result.understanding.hypothesis_proposals.size()) continue;

        const auto& original_hyp = result.understanding.hypothesis_proposals[idx];

        // Build a refiner callback: given residual feedback, produce a refined hypothesis
        // by re-running propose_hypotheses with the same evidence concepts
        auto refiner = [&](const HypothesisProposal& previous,
                           const std::string& /*residual_feedback*/) -> HypothesisProposal {
            // Re-generate hypotheses from the same evidence
            auto new_hyps = understanding.propose_hypotheses(
                previous.evidence_concepts, ltm, stm, context);

            if (!new_hyps.empty()) {
                return new_hyps[0];
            }
            return previous;  // Fallback: return unchanged
        };

        try {
            auto refinement_result = refinement_loop.run(original_hyp, refiner);

            // Replace the validation result with the refined one
            result.validated_hypotheses[idx] = refinement_result.final_validation;
            result.topology_c_refinements++;
        } catch (const std::exception& e) {
            std::cerr << "[ThinkingPipeline] Topology C refinement failed: "
                      << e.what() << "\n";
        }
    }
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
    ConceptModelRegistry& registry,
    EmbeddingManager& embeddings,
    UnderstandingLayer* understanding,
    KanValidator* kan_validator,
    GlobalDynamicsOperator* gdo,
    RefinementLoop* refinement_loop)
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

    // Build salient_ids for steps 7.5, 8, 8.5
    {
        std::unordered_set<ConceptId> seen;
        std::vector<ConceptId> salient_ids;
        for (auto cid : seed_concepts) {
            if (seen.insert(cid).second) salient_ids.push_back(cid);
        }
        for (auto& s : result.top_salient) {
            if (seen.insert(s.concept_id).second) salient_ids.push_back(s.concept_id);
        }

        // Step 7.5: KAN Graph Scan (Topology A — detect anomalies)
        if (config_.enable_topology_a) {
            result.kan_anomalies = step_kan_graph_scan(
                salient_ids, registry, embeddings, ltm);
        }

        // Step 8: UnderstandingLayer — pass ThoughtPaths for multi-hop reasoning
        if (config_.enable_understanding && understanding) {
            result.understanding = step_understanding(
                salient_ids, context, *understanding, cognitive, ltm, stm,
                result.best_paths);
        }
        result.steps_completed = 8;

        // Step 8.5: Topology A Investigation (KAN anomalies → hypotheses)
        if (config_.enable_topology_a && understanding && !result.kan_anomalies.empty()) {
            result.topology_a_hypotheses = step_topology_a(
                result.kan_anomalies, *understanding, ltm, stm, context);
            for (const auto& h : result.topology_a_hypotheses) {
                result.understanding.hypothesis_proposals.push_back(h);
            }
        }
    }

    // Step 9: KAN-LLM Validation — pass LTM for chain validation
    if (config_.enable_kan_validation && kan_validator &&
        !result.understanding.hypothesis_proposals.empty()) {
        result.validated_hypotheses = step_kan_validation(
            result.understanding.hypothesis_proposals, *kan_validator, &ltm);
    }
    result.steps_completed = 9;

    // Step 9C: Topology C Refinement
    if (config_.enable_topology_c && refinement_loop && understanding &&
        !result.validated_hypotheses.empty()) {
        step_topology_c(result, *refinement_loop, *understanding, ltm, stm, context);
    }

    // Step 9.5A: Trust ceiling enforcement (Topology B)
    for (auto& vr : result.validated_hypotheses) {
        double ceiling = vr.validated
            ? HypothesisProposal::KAN_VALIDATED_TRUST_CEILING
            : HypothesisProposal::LLM_ONLY_TRUST_CEILING;
        double capped = std::min(vr.assessment.metadata.trust, ceiling);
        EpistemicType type = vr.validated
            ? vr.assessment.metadata.type
            : EpistemicType::SPECULATION;
        vr.assessment.metadata = EpistemicMetadata(
            type, vr.assessment.metadata.status, capped);
    }

    // Step 9.5B: Feed validation back to ConceptPatternEngines
    if (understanding && !result.validated_hypotheses.empty()) {
        understanding->for_each_mini_llm([&](MiniLLM& llm) {
            auto* kan_aware = dynamic_cast<ConceptPatternEngine*>(&llm);
            if (kan_aware) {
                kan_aware->train_from_validation(result.validated_hypotheses);
            }
        });
    }

    // Step 10: Complete
    result.steps_completed = 10;
    auto end = std::chrono::steady_clock::now();
    result.total_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

} // namespace brain19
