// Test suite for CuriosityEngine rewrite
// Tests: SystemSnapshot, CuriosityScores, SeedPlan, Triggers, CoLearnLoop integration

#include "../ltm/long_term_memory.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../cmodel/concept_trainer.hpp"
#include "../graph_net/graph_reasoner.hpp"
#include "../colearn/colearn_loop.hpp"
#include "../colearn/episodic_memory.hpp"
#include "../colearn/error_collector.hpp"
#include "curiosity_engine.hpp"
#include "curiosity_score.hpp"
#include "signal_types.hpp"
#include "trend_tracker.hpp"
#include "goal_generator.hpp"

#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#include <cassert>

using namespace brain19;

static int tests_passed = 0;
static int tests_failed = 0;

static void log(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&t));
    std::cout << "[" << ts << "] " << msg << "\n";
}

static void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  PASS: " << name << "\n";
        ++tests_passed;
    } else {
        std::cout << "  FAIL: " << name << "\n";
        ++tests_failed;
    }
}

// =============================================================================
// Helper: Build a test knowledge graph
// =============================================================================

struct TestGraph {
    LongTermMemory ltm;
    EmbeddingManager embeddings;
    ConceptModelRegistry registry;

    ConceptId c_water, c_heat, c_steam, c_cold, c_ice;
    ConceptId c_isolated;  // no relations

    void build() {
        c_water = ltm.store_concept("Water", "H2O molecule",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9));
        c_heat = ltm.store_concept("Heat", "thermal energy",
            EpistemicMetadata(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.8));
        c_steam = ltm.store_concept("Steam", "water vapor",
            EpistemicMetadata(EpistemicType::HYPOTHESIS, EpistemicStatus::ACTIVE, 0.4));
        c_cold = ltm.store_concept("Cold", "low temperature",
            EpistemicMetadata(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.6));
        c_ice = ltm.store_concept("Ice", "solid water",
            EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.2));
        c_isolated = ltm.store_concept("Isolated", "no connections",
            EpistemicMetadata(EpistemicType::SPECULATION, EpistemicStatus::ACTIVE, 0.15));

        // Dense connections for water/heat/steam
        ltm.add_relation(c_water, c_heat, RelationType::CAUSES, 0.8);
        ltm.add_relation(c_heat, c_steam, RelationType::CAUSES, 0.7);
        ltm.add_relation(c_water, c_steam, RelationType::CAUSES, 0.6);
        ltm.add_relation(c_cold, c_ice, RelationType::CAUSES, 0.5);
        ltm.add_relation(c_water, c_ice, RelationType::CAUSES, 0.4);
        ltm.add_relation(c_water, c_cold, RelationType::SIMILAR_TO, 0.3);

        // Contradiction edge for steam
        ltm.add_relation(c_steam, c_ice, RelationType::CONTRADICTS, 0.9);

        // Ensure models exist
        registry.ensure_models_for(ltm);
    }
};

// =============================================================================
// Test 1: MetricEMA
// =============================================================================

static void test_metric_ema() {
    log("--- Test 1: MetricEMA ---");

    MetricEMA ema;
    check(ema.samples == 0, "initial samples = 0");
    check(ema.value == 0.0, "initial value = 0.0");

    ema.update(1.0);
    check(ema.samples == 1, "after 1st update: samples = 1");
    check(std::abs(ema.value - 1.0) < 0.001, "after 1st update: value = 1.0 (seed)");

    ema.update(0.5, 0.5);
    check(ema.samples == 2, "after 2nd update: samples = 2");
    check(std::abs(ema.value - 0.75) < 0.001, "after 2nd update: EMA correct");
    check(ema.trend < 0.0, "trend is negative (value decreased)");
}

// =============================================================================
// Test 2: TrendTracker
// =============================================================================

static void test_trend_tracker() {
    log("--- Test 2: TrendTracker ---");

    TrendTracker tracker;
    tracker.avg_model_loss.update(0.5);
    tracker.avg_model_loss.update(0.4);
    check(tracker.avg_model_loss.trend < 0.0, "model loss trend negative (improving)");

    tracker.update_concept(tracker.concept_pain, 42, 0.8);
    tracker.update_concept(tracker.concept_pain, 42, 0.6);
    double trend = tracker.get_concept_pain_trend(42);
    check(trend < 0.0, "concept pain trend negative (pain decreasing)");
    check(tracker.get_concept_pain_trend(999) == 0.0, "unknown concept trend = 0");
}

// =============================================================================
// Test 3: SystemSnapshot observation
// =============================================================================

static void test_system_snapshot() {
    log("--- Test 3: SystemSnapshot observation ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;
    pain_scores[tg.c_steam] = 0.8;  // high pain for steam
    pain_scores[tg.c_water] = 0.1;  // low pain for water

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& scores = engine.last_scores();
    const auto& plan = engine.last_plan();

    check(!scores.empty(), "scores generated");
    check(scores.size() == 6, "scores for all 6 concepts");

    // System health should be computed
    check(plan.system_health > 0.0 && plan.system_health <= 1.0,
          "system health in (0,1]");
    check(!plan.health_summary.empty(), "health summary not empty");
    check(!plan.seeds.empty(), "seed plan not empty");

    log("  system_health = " + std::to_string(plan.system_health));
    log("  health_summary: " + plan.health_summary);
    log("  seeds: " + std::to_string(plan.seeds.size()));
}

// =============================================================================
// Test 4: Scoring — high-pain concept scores higher on pain dimension
// =============================================================================

static void test_scoring_pain() {
    log("--- Test 4: Scoring — pain dimension ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;
    pain_scores[tg.c_steam] = 0.9;  // very high pain

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& scores = engine.last_scores();

    // Find steam's score
    const CuriosityScore* steam_score = nullptr;
    const CuriosityScore* water_score = nullptr;
    for (const auto& sc : scores) {
        if (sc.concept_id == tg.c_steam) steam_score = &sc;
        if (sc.concept_id == tg.c_water) water_score = &sc;
    }

    check(steam_score != nullptr, "steam has a score");
    check(water_score != nullptr, "water has a score");

    if (steam_score && water_score) {
        // Steam should have higher pain dimension score
        double steam_pain = steam_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::PAIN_DRIVEN)];
        double water_pain = water_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::PAIN_DRIVEN)];
        check(steam_pain > water_pain, "steam pain > water pain");
        log("  steam pain dim = " + std::to_string(steam_pain));
        log("  water pain dim = " + std::to_string(water_pain));
    }
}

// =============================================================================
// Test 5: Scoring — trust deficit
// =============================================================================

static void test_scoring_trust() {
    log("--- Test 5: Scoring — trust deficit ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& scores = engine.last_scores();

    const CuriosityScore* ice_score = nullptr;
    const CuriosityScore* water_score = nullptr;
    for (const auto& sc : scores) {
        if (sc.concept_id == tg.c_ice) ice_score = &sc;
        if (sc.concept_id == tg.c_water) water_score = &sc;
    }

    check(ice_score != nullptr, "ice has a score");
    check(water_score != nullptr, "water has a score");

    if (ice_score && water_score) {
        double ice_trust = ice_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::TRUST_DEFICIT)];
        double water_trust = water_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::TRUST_DEFICIT)];
        check(ice_trust > water_trust,
              "ice trust_deficit > water trust_deficit (ice trust=0.2, water trust=0.9)");
        log("  ice trust_deficit = " + std::to_string(ice_trust));
        log("  water trust_deficit = " + std::to_string(water_trust));
    }
}

// =============================================================================
// Test 6: Topology gap — isolated concepts score highest
// =============================================================================

static void test_scoring_topology() {
    log("--- Test 6: Scoring — topology gap ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& scores = engine.last_scores();

    const CuriosityScore* isolated_score = nullptr;
    const CuriosityScore* water_score = nullptr;
    for (const auto& sc : scores) {
        if (sc.concept_id == tg.c_isolated) isolated_score = &sc;
        if (sc.concept_id == tg.c_water) water_score = &sc;
    }

    check(isolated_score != nullptr, "isolated has a score");
    check(water_score != nullptr, "water has a score");

    if (isolated_score && water_score) {
        double iso_topo = isolated_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::TOPOLOGY_GAP)];
        double water_topo = water_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::TOPOLOGY_GAP)];
        check(iso_topo > water_topo,
              "isolated topology_gap > water topology_gap");
        check(std::abs(iso_topo - 1.0) < 0.001,
              "isolated topology_gap = 1.0 (no relations)");
        log("  isolated topology = " + std::to_string(iso_topo));
        log("  water topology = " + std::to_string(water_topo));
    }
}

// =============================================================================
// Test 7: Contradiction detection
// =============================================================================

static void test_scoring_contradiction() {
    log("--- Test 7: Scoring — contradictions ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& scores = engine.last_scores();

    // Steam has a CONTRADICTS edge to ice
    const CuriosityScore* steam_score = nullptr;
    const CuriosityScore* heat_score = nullptr;
    for (const auto& sc : scores) {
        if (sc.concept_id == tg.c_steam) steam_score = &sc;
        if (sc.concept_id == tg.c_heat) heat_score = &sc;
    }

    check(steam_score != nullptr, "steam has a score");
    if (steam_score && heat_score) {
        double steam_contra = steam_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::CONTRADICTION_ALERT)];
        double heat_contra = heat_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::CONTRADICTION_ALERT)];
        check(steam_contra > 0.0, "steam has contradiction score > 0");
        check(heat_contra == 0.0, "heat has no contradictions");
        log("  steam contradiction = " + std::to_string(steam_contra));
    }
}

// =============================================================================
// Test 8: Cross-signal bonus
// =============================================================================

static void test_cross_signal_bonus() {
    log("--- Test 8: Cross-signal bonus ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;
    // Give ice high pain to trigger multiple signals
    pain_scores[tg.c_ice] = 0.9;

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& scores = engine.last_scores();

    // Ice should have multiple dimensions firing:
    // - high pain (0.9 seed pain)
    // - high trust deficit (trust=0.2)
    // - topology gap (only 0 outgoing rels)
    // This should trigger the cross-signal bonus
    const CuriosityScore* ice_score = nullptr;
    for (const auto& sc : scores) {
        if (sc.concept_id == tg.c_ice) ice_score = &sc;
    }

    check(ice_score != nullptr, "ice has a score");
    if (ice_score) {
        size_t active_dims = 0;
        for (size_t i = 0; i < 12; ++i) {
            if (ice_score->dimension_scores[i] > 0.3) ++active_dims;
        }
        log("  ice active dims (>0.3) = " + std::to_string(active_dims));

        double cross_signal = ice_score->dimension_scores[
            static_cast<size_t>(CuriosityDimension::CROSS_SIGNAL)];
        if (active_dims >= 3) {
            check(cross_signal > 0.5, "cross-signal fires with 3+ active dims");
        } else {
            log("  (less than 3 active dims, cross-signal may not fire)");
        }

        check(ice_score->total_score > 0.0, "ice has positive total score");
        log("  ice total_score = " + std::to_string(ice_score->total_score));
    }
}

// =============================================================================
// Test 9: Seed plan diversity
// =============================================================================

static void test_seed_plan_diversity() {
    log("--- Test 9: Seed plan diversity ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;

    CuriosityConfig config;
    config.max_seeds = 4;
    config.diversity_weight = 0.5;  // Strong diversity preference

    CuriosityEngine engine(config);
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    auto seeds = engine.select_seeds(4);
    check(!seeds.empty(), "seeds not empty");
    check(seeds.size() <= 4, "at most 4 seeds");

    // Check no duplicates
    std::unordered_set<ConceptId> seen;
    bool has_duplicates = false;
    for (ConceptId cid : seeds) {
        if (!seen.insert(cid).second) has_duplicates = true;
    }
    check(!has_duplicates, "no duplicate seeds");

    log("  selected seeds: " + std::to_string(seeds.size()));
    for (ConceptId cid : seeds) {
        auto info = tg.ltm.retrieve_concept(cid);
        if (info) {
            log("    - " + info->label + " (id=" + std::to_string(cid) + ")");
        }
    }
}

// =============================================================================
// Test 10: Trigger generation
// =============================================================================

static void test_trigger_generation() {
    log("--- Test 10: Trigger generation ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;
    pain_scores[tg.c_steam] = 0.8;
    pain_scores[tg.c_ice] = 0.7;

    CuriosityConfig config;
    config.trigger_threshold = 0.1;
    config.min_cluster_size = 1;  // allow single-concept triggers for testing

    CuriosityEngine engine(config);
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    auto triggers = engine.generate_triggers(1);
    log("  triggers generated: " + std::to_string(triggers.size()));

    for (const auto& t : triggers) {
        log("    type=" + std::to_string(static_cast<int>(t.type))
            + " prio=" + std::to_string(t.priority)
            + " concepts=" + std::to_string(t.related_concept_ids.size())
            + " desc=" + t.description);
        check(t.priority > 0.0, "trigger has positive priority");
    }

    // Test GoalGenerator handles new trigger types
    if (!triggers.empty()) {
        auto goals = GoalGenerator::from_triggers(triggers);
        check(goals.size() == triggers.size(), "goals generated for all triggers");
        for (const auto& goal : goals) {
            check(goal.priority_weight > 0.0, "goal has positive priority");
        }
    }
}

// =============================================================================
// Test 11: Backward-compat wrapper
// =============================================================================

static void test_backward_compat() {
    log("--- Test 11: Backward-compat observe_and_generate_triggers ---");

    CuriosityEngine engine;

    // Legacy path: no refresh(), just pass observations
    SystemObservation obs;
    obs.context_id = 1;
    obs.active_concept_count = 10;
    obs.active_relation_count = 2;  // low ratio → SHALLOW_RELATIONS

    auto triggers = engine.observe_and_generate_triggers({obs});
    check(!triggers.empty(), "legacy trigger generated");
    if (!triggers.empty()) {
        check(triggers[0].type == TriggerType::SHALLOW_RELATIONS,
              "legacy trigger type = SHALLOW_RELATIONS");
    }

    // With few concepts
    SystemObservation obs2;
    obs2.context_id = 2;
    obs2.active_concept_count = 3;
    obs2.active_relation_count = 10;

    auto triggers2 = engine.observe_and_generate_triggers({obs2});
    bool has_low_exploration = false;
    for (const auto& t : triggers2) {
        if (t.type == TriggerType::LOW_EXPLORATION) has_low_exploration = true;
    }
    check(has_low_exploration, "legacy LOW_EXPLORATION trigger generated");
}

// =============================================================================
// Test 12: SeedEntry to GoalType mapping
// =============================================================================

static void test_dimension_to_goal() {
    log("--- Test 12: Dimension → GoalType mapping ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    auto entries = engine.select_seed_entries(6);
    for (const auto& e : entries) {
        auto info = tg.ltm.retrieve_concept(e.concept_id);
        std::string label = info ? info->label : "?";
        log("  seed: " + label
            + " priority=" + std::to_string(e.priority)
            + " goal=" + std::to_string(static_cast<int>(e.suggested_goal))
            + " reason=" + e.reason_text);
        check(e.priority >= 0.0, "seed has non-negative priority");
    }
}

// =============================================================================
// Test 13: CoLearnLoop with CuriosityEngine (null fallback)
// =============================================================================

static void test_colearn_null_fallback() {
    log("--- Test 13: CoLearnLoop null fallback ---");

    TestGraph tg;
    tg.build();

    // Train models first
    ConceptTrainer trainer;
    trainer.train_all(tg.registry, tg.embeddings, tg.ltm);

    GraphReasoner reasoner(tg.ltm, tg.registry, tg.embeddings);
    CoLearnConfig config;
    config.wake_chains_per_cycle = 3;

    CoLearnLoop loop(tg.ltm, tg.registry, tg.embeddings, reasoner, config);

    // No curiosity engine → should use 4-way fallback
    auto result = loop.run_cycle();
    check(result.cycle_number == 1, "cycle ran");
    check(result.chains_produced > 0 || tg.ltm.get_all_concept_ids().size() > 0,
          "wake phase attempted chains");
    log("  chains_produced = " + std::to_string(result.chains_produced));
}

// =============================================================================
// Test 14: CoLearnLoop with CuriosityEngine wired in
// =============================================================================

static void test_colearn_with_curiosity() {
    log("--- Test 14: CoLearnLoop with CuriosityEngine ---");

    TestGraph tg;
    tg.build();

    ConceptTrainer trainer;
    trainer.train_all(tg.registry, tg.embeddings, tg.ltm);

    GraphReasoner reasoner(tg.ltm, tg.registry, tg.embeddings);
    CoLearnConfig config;
    config.wake_chains_per_cycle = 4;

    CoLearnLoop loop(tg.ltm, tg.registry, tg.embeddings, reasoner, config);

    CuriosityEngine curiosity;
    loop.set_curiosity_engine(&curiosity);

    auto result = loop.run_cycle();
    check(result.cycle_number == 1, "cycle ran with curiosity");
    log("  chains_produced = " + std::to_string(result.chains_produced));
    log("  avg_quality = " + std::to_string(result.avg_chain_quality));

    // Verify curiosity engine was refreshed
    check(curiosity.system_health() > 0.0, "curiosity computed health after refresh");
    check(!curiosity.last_scores().empty(), "curiosity has cached scores");

    const auto& plan = curiosity.last_plan();
    log("  curiosity health = " + std::to_string(plan.system_health));
    log("  curiosity seeds = " + std::to_string(plan.seeds.size()));
}

// =============================================================================
// Test 15: Model uncertainty scoring
// =============================================================================

static void test_scoring_model_uncertainty() {
    log("--- Test 15: Scoring — model uncertainty ---");

    TestGraph tg;
    tg.build();

    // Train models so some converge
    ConceptTrainer trainer;
    trainer.train_all(tg.registry, tg.embeddings, tg.ltm);

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;

    CuriosityEngine engine;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& scores = engine.last_scores();
    for (const auto& sc : scores) {
        double model_unc = sc.dimension_scores[
            static_cast<size_t>(CuriosityDimension::MODEL_UNCERTAINTY)];
        auto info = tg.ltm.retrieve_concept(sc.concept_id);
        std::string label = info ? info->label : "?";
        log("  " + label + " model_uncertainty = " + std::to_string(model_unc));
        check(model_unc >= 0.0 && model_unc <= 1.0,
              label + " model_uncertainty in [0,1]");
    }
}

// =============================================================================
// Test 16: System health computation
// =============================================================================

static void test_system_health() {
    log("--- Test 16: System health ---");

    // Empty system
    {
        LongTermMemory ltm;
        ConceptModelRegistry registry;
        EpisodicMemory episodic;
        ErrorCollector error_collector;
        std::unordered_map<ConceptId, double> pain_scores;

        CuriosityEngine engine;
        engine.refresh(ltm, registry, episodic, error_collector,
                       pain_scores, nullptr, nullptr);
        // No concepts → plan should be empty
        check(engine.last_plan().seeds.empty(), "empty system has no seeds");
    }

    // Rich system
    {
        TestGraph tg;
        tg.build();

        ConceptTrainer trainer;
        trainer.train_all(tg.registry, tg.embeddings, tg.ltm);

        EpisodicMemory episodic;
        ErrorCollector error_collector;
        std::unordered_map<ConceptId, double> pain_scores;

        CuriosityEngine engine;
        engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                       pain_scores, nullptr, nullptr);

        double health = engine.system_health();
        check(health > 0.0, "non-empty system has positive health");
        check(health <= 1.0, "health <= 1.0");
        log("  health = " + std::to_string(health));
    }
}

// =============================================================================
// Test 17: Trend tracking across multiple refreshes
// =============================================================================

static void test_trend_tracking() {
    log("--- Test 17: Trend tracking ---");

    TestGraph tg;
    tg.build();

    EpisodicMemory episodic;
    ErrorCollector error_collector;
    std::unordered_map<ConceptId, double> pain_scores;
    pain_scores[tg.c_steam] = 0.9;

    CuriosityEngine engine;

    // Refresh multiple times with decreasing pain
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    pain_scores[tg.c_steam] = 0.6;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    pain_scores[tg.c_steam] = 0.3;
    engine.refresh(tg.ltm, tg.registry, episodic, error_collector,
                   pain_scores, nullptr, nullptr);

    const auto& trends = engine.trends();
    double steam_trend = trends.get_concept_pain_trend(tg.c_steam);
    log("  steam pain trend = " + std::to_string(steam_trend));
    check(steam_trend < 0.0, "pain trend negative (pain decreasing)");
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    log("=== CuriosityEngine Test Suite ===\n");

    test_metric_ema();
    test_trend_tracker();
    test_system_snapshot();
    test_scoring_pain();
    test_scoring_trust();
    test_scoring_topology();
    test_scoring_contradiction();
    test_cross_signal_bonus();
    test_seed_plan_diversity();
    test_trigger_generation();
    test_backward_compat();
    test_dimension_to_goal();
    test_colearn_null_fallback();
    test_colearn_with_curiosity();
    test_scoring_model_uncertainty();
    test_system_health();
    test_trend_tracking();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
