"""Brain19 Pipeline Step Templates.

Each template represents a real subsystem call from ThinkingPipeline::execute()
(backend/core/thinking_pipeline.cpp). Templates carry the C++ code fragments,
required includes, and configurable parameters needed to assemble a complete
main.cpp.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemplateParam:
    name: str
    type: str  # "double", "size_t", "bool", "string"
    default: str
    description: str


@dataclass
class StepTemplate:
    id: str
    label: str
    subsystem: str
    category: str
    step: str  # "1", "2", "2.5", "3", "4-5", "6", "7", "8", "9", "BG"
    description: str
    includes: list[str] = field(default_factory=list)
    init_code: str = ""
    exec_code: str = ""
    params: list[TemplateParam] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # step IDs this requires
    optional: bool = False


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, StepTemplate] = {}


def _register(t: StepTemplate) -> None:
    TEMPLATES[t.id] = t


# ── Step 1: Seed Activation ──────────────────────────────────────────────────

_register(StepTemplate(
    id="stm",
    label="Seed Activation",
    subsystem="STM + BrainController",
    category="memory",
    step="1",
    description="Activates seed concepts in Short-Term Memory with initial activation value. Entry point for the thinking cycle.",
    includes=[
        '"memory/stm.hpp"',
        '"memory/brain_controller.hpp"',
    ],
    init_code="",  # brain/stm already initialized by orchestrator
    exec_code="""\
    // Step 1: Activate seed concepts in STM
    for (auto cid : seed_concepts) {
        brain.activate_concept_in_context(
            ctx, cid, {{initial_activation}}, ActivationClass::CORE_KNOWLEDGE);
    }
    result.steps_completed = 1;""",
    params=[
        TemplateParam("initial_activation", "double", "0.8",
                       "Initial activation energy for seed concepts"),
    ],
))

# ── Step 2: Spreading Activation ─────────────────────────────────────────────

_register(StepTemplate(
    id="spreading",
    label="Spreading Activation",
    subsystem="CognitiveDynamics",
    category="cognitive",
    step="2",
    description="Spreads activation from seeds through the knowledge graph. Energy decays with distance. Multi-hop propagation.",
    includes=[
        '"cognitive/cognitive_dynamics.hpp"',
    ],
    exec_code="""\
    // Step 2: Spreading Activation
    auto spread_stats = cognitive.spread_activation_multi(
        seed_concepts, {{initial_activation}}, ctx, ltm, stm);
    result.steps_completed = 2;""",
    params=[
        TemplateParam("initial_activation", "double", "0.8",
                       "Activation energy for spreading"),
    ],
    dependencies=["stm"],
))

# ── Step 2.5: FocusCursor ────────────────────────────────────────────────────

_register(StepTemplate(
    id="cursor",
    label="FocusCursor",
    subsystem="FocusCursorManager",
    category="traversal",
    step="2.5",
    description="Goal-directed graph traversal. Seeds augmented by GDO top-3 activations. Results fed back to GDO.",
    includes=[
        '"cursor/focus_cursor_manager.hpp"',
        '"cursor/goal_state.hpp"',
        '"cursor/traversal_types.hpp"',
    ],
    exec_code="""\
    // Step 2.5: FocusCursor traversal
    if (config.enable_focus_cursor) {
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
        FocusCursorManager mgr(ltm, registry, embeddings, stm, config.cursor_config);
        Vec10 query_context{};
        auto qr = mgr.process_seeds(cursor_seeds, query_context, default_goal);
        if (!qr.best_chain.empty()) {
            result.cursor_result = qr.best_chain;
            if (gdo) gdo->feed_traversal_result(qr.best_chain);
            mgr.persist_to_stm(ctx, qr.best_chain);
        }
    }""",
    params=[
        TemplateParam("enable_focus_cursor", "bool", "true",
                       "Enable goal-directed FocusCursor traversal"),
    ],
    dependencies=["spreading"],
    optional=True,
))

# ── Step 3: Salience & Focus ─────────────────────────────────────────────────

_register(StepTemplate(
    id="salience",
    label="Salience & Focus",
    subsystem="CognitiveDynamics",
    category="cognitive",
    step="3",
    description="Computes salience scores for active concepts. Initializes cognitive focus on top-k salient concepts.",
    includes=[
        '"cognitive/cognitive_dynamics.hpp"',
    ],
    exec_code="""\
    // Gather active concepts
    result.activated_concepts = stm.get_active_concepts(ctx, 0.05);

    // Step 3: Compute Salience + Focus
    result.top_salient = cognitive.get_top_k_salient(
        result.activated_concepts, {{top_k_salient}}, ctx, ltm, stm);
    cognitive.init_focus(ctx);
    for (auto& s : result.top_salient) {
        cognitive.focus_on(ctx, s.concept_id, s.salience);
    }
    result.steps_completed = 3;""",
    params=[
        TemplateParam("top_k_salient", "size_t", "10",
                       "Number of top-k salient concepts to focus on"),
    ],
    dependencies=["spreading"],
))

# ── Step 4-5: RelevanceMaps ──────────────────────────────────────────────────

_register(StepTemplate(
    id="relevance",
    label="RelevanceMaps",
    subsystem="MicroModels",
    category="micromodel",
    step="4-5",
    description="Generates bilinear relevance maps per salient concept, then combines via weighted overlay for creative associations.",
    includes=[
        '"micromodel/relevance_map.hpp"',
        '"micromodel/micro_model_registry.hpp"',
        '"micromodel/embedding_manager.hpp"',
    ],
    exec_code="""\
    // Step 4-5: Generate and combine RelevanceMaps
    {
        std::vector<RelevanceMap> maps;
        size_t count = std::min(result.top_salient.size(), (size_t){{max_relevance_maps}});
        for (size_t i = 0; i < count; ++i) {
            auto cid = result.top_salient[i].concept_id;
            if (registry.has_model(cid)) {
                auto map = RelevanceMap::compute(
                    cid, registry, embeddings, ltm,
                    RelationType::IS_A, "query");
                maps.push_back(std::move(map));
            }
        }
        if (maps.size() == 1) {
            result.combined_relevance = std::move(maps[0]);
        } else if (maps.size() > 1) {
            result.combined_relevance = RelevanceMap::combine(maps, OverlayMode::WEIGHTED_AVERAGE);
        }
    }
    result.steps_completed = 5;""",
    params=[
        TemplateParam("max_relevance_maps", "size_t", "5",
                       "Maximum number of relevance maps to generate and combine"),
    ],
    dependencies=["salience"],
))

# ── Step 6: ThoughtPaths ─────────────────────────────────────────────────────

_register(StepTemplate(
    id="paths",
    label="ThoughtPaths",
    subsystem="CognitiveDynamics",
    category="cognitive",
    step="6",
    description="Finds best thought paths from each seed concept through the activation landscape. Sorted by path score.",
    includes=[
        '"cognitive/cognitive_dynamics.hpp"',
    ],
    exec_code="""\
    // Step 6: Find ThoughtPaths
    {
        std::vector<ThoughtPath> all_paths;
        for (auto cid : seed_concepts) {
            auto paths = cognitive.find_best_paths(cid, ctx, ltm, stm);
            for (auto& p : paths) {
                all_paths.push_back(std::move(p));
            }
        }
        std::sort(all_paths.begin(), all_paths.end());
        if (all_paths.size() > {{max_paths}}) {
            all_paths.resize({{max_paths}});
        }
        result.best_paths = std::move(all_paths);
    }
    result.steps_completed = 6;""",
    params=[
        TemplateParam("max_paths", "size_t", "20",
                       "Maximum number of thought paths to keep"),
    ],
    dependencies=["salience"],
))

# ── Step 7: CuriosityEngine ──────────────────────────────────────────────────

_register(StepTemplate(
    id="curiosity",
    label="CuriosityEngine",
    subsystem="CuriosityEngine + GoalGenerator",
    category="curiosity",
    step="7",
    description="Observes STM state for knowledge gaps. Generates triggers (SHALLOW_RELATIONS, MISSING_DEPTH, etc.) and converts to GoalStates.",
    includes=[
        '"curiosity/curiosity_engine.hpp"',
        '"curiosity/goal_generator.hpp"',
    ],
    exec_code="""\
    // Step 7: CuriosityEngine
    if ({{enable_curiosity}}) {
        SystemObservation obs;
        obs.context_id = ctx;
        obs.active_concept_count = stm.debug_active_concept_count(ctx);
        obs.active_relation_count = stm.debug_active_relation_count(ctx);
        result.curiosity_triggers = curiosity.observe_and_generate_triggers({obs});
        result.generated_goals = GoalGenerator::from_triggers(result.curiosity_triggers);
    }
    result.steps_completed = 7;""",
    params=[
        TemplateParam("enable_curiosity", "bool", "true",
                       "Enable curiosity-driven gap detection"),
    ],
    dependencies=["paths"],
    optional=True,
))

# ── Step 8: UnderstandingLayer ────────────────────────────────────────────────

_register(StepTemplate(
    id="understanding",
    label="UnderstandingLayer",
    subsystem="MiniLLMs",
    category="understanding",
    step="8",
    description="Runs MiniLLM understanding cycle on top salient concept. Generates hypothesis proposals for validation.",
    includes=[
        '"understanding/understanding_layer.hpp"',
    ],
    exec_code="""\
    // Step 8: UnderstandingLayer
    if ({{enable_understanding}} && understanding) {
        std::vector<ConceptId> salient_ids;
        salient_ids.reserve(result.top_salient.size());
        for (auto& s : result.top_salient) {
            salient_ids.push_back(s.concept_id);
        }
        if (!salient_ids.empty()) {
            result.understanding = understanding->perform_understanding_cycle(
                salient_ids[0], cognitive, ltm, stm, ctx);
        }
    }
    result.steps_completed = 8;""",
    params=[
        TemplateParam("enable_understanding", "bool", "true",
                       "Enable MiniLLM understanding cycle"),
    ],
    dependencies=["curiosity"],
    optional=True,
))

# ── Step 9: KAN Validation ───────────────────────────────────────────────────

_register(StepTemplate(
    id="kan",
    label="KAN Validation",
    subsystem="KanValidator",
    category="hybrid",
    step="9",
    description="Validates hypothesis proposals using Kolmogorov-Arnold Networks. Provides confidence scores and explanations.",
    includes=[
        '"hybrid/kan_validator.hpp"',
    ],
    exec_code="""\
    // Step 9: KAN-LLM Validation
    if ({{enable_kan_validation}} && kan_validator &&
        !result.understanding.hypothesis_proposals.empty()) {
        for (auto& hyp : result.understanding.hypothesis_proposals) {
            try {
                result.validated_hypotheses.push_back(kan_validator->validate(hyp));
            } catch (const std::exception& e) {
                std::cerr << "[Pipeline] KAN validation failed: " << e.what() << "\\n";
            }
        }
    }
    result.steps_completed = 9;""",
    params=[
        TemplateParam("enable_kan_validation", "bool", "true",
                       "Enable KAN hypothesis validation"),
    ],
    dependencies=["understanding"],
    optional=True,
))

# ── Background: GlobalDynamicsOperator ────────────────────────────────────────

_register(StepTemplate(
    id="gdo",
    label="Global Dynamics",
    subsystem="GlobalDynamicsOperator",
    category="background",
    step="BG",
    description="Background thread. Ticks every 500ms: decay activations, prune, maybe trigger autonomous thinking when energy > 30.",
    includes=[
        '"cognitive/global_dynamics_operator.hpp"',
    ],
    init_code="""\
    // GDO background thread initialization
    GlobalDynamicsOperator gdo_instance(gdo_config);
    gdo = &gdo_instance;
    gdo->start();""",
    exec_code="",  # runs as background thread, not a pipeline step
    params=[
        TemplateParam("enable_gdo", "bool", "false",
                       "Enable Global Dynamics Operator background thread"),
        TemplateParam("gdo_tick_interval_ms", "size_t", "500",
                       "GDO tick interval in milliseconds"),
    ],
    optional=True,
))


def get_all_templates() -> list[dict]:
    """Return all templates as serializable dicts."""
    result = []
    for t in TEMPLATES.values():
        result.append({
            "id": t.id,
            "label": t.label,
            "subsystem": t.subsystem,
            "category": t.category,
            "step": t.step,
            "description": t.description,
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "description": p.description,
                }
                for p in t.params
            ],
            "dependencies": t.dependencies,
            "optional": t.optional,
        })
    return result


def get_template(step_id: str) -> StepTemplate | None:
    return TEMPLATES.get(step_id)
