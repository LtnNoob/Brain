#include "global_dynamics_operator.hpp"
#include "../memory/active_relation.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>

using namespace brain19;

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { \
    tests_total++; \
    std::cout << "  TEST " << tests_total << ": " << name << "... "; \
} while(0)

#define PASS() do { \
    tests_passed++; \
    std::cout << "PASS" << std::endl; \
} while(0)

void test_construct_default() {
    TEST("Default construction");
    GlobalDynamicsOperator gdo;
    assert(!gdo.is_running());
    assert(gdo.get_global_energy() == 0.0);
    PASS();
}

void test_inject_energy() {
    TEST("Inject energy");
    GlobalDynamicsOperator gdo;
    gdo.inject_energy(10.0);
    assert(gdo.get_global_energy() == 10.0);
    gdo.inject_energy(5.0);
    assert(gdo.get_global_energy() == 15.0);
    PASS();
}

void test_inject_energy_cap() {
    TEST("Energy cap at max");
    GDOConfig cfg;
    cfg.max_global_energy = 20.0;
    GlobalDynamicsOperator gdo(cfg);
    gdo.inject_energy(100.0);
    assert(gdo.get_global_energy() == 20.0);
    PASS();
}

void test_inject_seeds() {
    TEST("Inject seeds");
    GlobalDynamicsOperator gdo;
    gdo.inject_seeds({1, 2, 3}, 0.8);
    auto snap = gdo.get_activation_snapshot(10);
    assert(snap.size() == 3);
    for (auto& [cid, val] : snap) {
        assert(val == 0.8);
    }
    PASS();
}

void test_start_stop() {
    TEST("Start and stop");
    GDOConfig cfg;
    cfg.tick_interval = std::chrono::milliseconds(50);
    GlobalDynamicsOperator gdo(cfg);
    gdo.start();
    assert(gdo.is_running());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gdo.stop();
    assert(!gdo.is_running());
    PASS();
}

void test_decay() {
    TEST("Decay reduces activations");
    GDOConfig cfg;
    cfg.tick_interval = std::chrono::milliseconds(20);
    cfg.decay_rate = 0.5;  // aggressive decay for testing
    cfg.enable_autonomous_thinking = false;
    GlobalDynamicsOperator gdo(cfg);

    gdo.inject_seeds({42}, 1.0);
    double before = gdo.get_activation_snapshot(1)[0].second;

    gdo.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gdo.stop();

    auto snap = gdo.get_activation_snapshot(1);
    if (!snap.empty()) {
        assert(snap[0].second < before);
    }
    // Activation should have decayed significantly
    PASS();
}

void test_snapshot() {
    TEST("Get snapshot");
    GlobalDynamicsOperator gdo;
    gdo.inject_seeds({10, 20, 30}, 0.5);
    gdo.inject_energy(7.0);

    auto snap = gdo.get_snapshot(5);
    assert(snap.active_concepts == 3);
    assert(snap.global_energy == 8.5);  // 7.0 + 0.5*3
    assert(snap.top_activations.size() == 3);
    PASS();
}

void test_thinking_callback() {
    TEST("Thinking callback fires");
    GDOConfig cfg;
    cfg.tick_interval = std::chrono::milliseconds(20);
    cfg.thinking_trigger_energy = 5.0;
    cfg.enable_autonomous_thinking = true;
    GlobalDynamicsOperator gdo(cfg);

    std::atomic<int> callback_count{0};
    std::vector<ConceptId> received_seeds;
    std::mutex seed_mtx;

    gdo.set_thinking_callback([&](const std::vector<ConceptId>& seeds) {
        callback_count++;
        std::lock_guard<std::mutex> lock(seed_mtx);
        received_seeds = seeds;
    });

    gdo.inject_seeds({100, 200}, 0.9);
    gdo.inject_energy(10.0);

    gdo.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    gdo.stop();

    assert(callback_count.load() >= 1);
    {
        std::lock_guard<std::mutex> lock(seed_mtx);
        assert(!received_seeds.empty());
    }
    PASS();
}

void test_no_thinking_below_threshold() {
    TEST("No thinking below energy threshold");
    GDOConfig cfg;
    cfg.tick_interval = std::chrono::milliseconds(20);
    cfg.thinking_trigger_energy = 100.0;  // very high
    cfg.enable_autonomous_thinking = true;
    GlobalDynamicsOperator gdo(cfg);

    std::atomic<int> callback_count{0};
    gdo.set_thinking_callback([&](const std::vector<ConceptId>&) {
        callback_count++;
    });

    gdo.inject_energy(5.0);  // well below threshold
    gdo.inject_seeds({1}, 0.5);

    gdo.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    gdo.stop();

    assert(callback_count.load() == 0);
    PASS();
}

void test_feed_traversal() {
    TEST("Feed traversal result");
    GlobalDynamicsOperator gdo;

    TraversalResult result;
    result.chain_score = 0.8;
    result.chain.push_back({.concept_id = 1, .relation_from = RelationType::IS_A,
                            .weight_at_entry = 0.9, .context_at_entry = {}, .depth = 0});
    result.chain.push_back({.concept_id = 2, .relation_from = RelationType::CAUSES,
                            .weight_at_entry = 0.7, .context_at_entry = {}, .depth = 1});
    result.concept_sequence = {1, 2};
    result.total_steps = 2;

    gdo.feed_traversal_result(result);

    auto snap = gdo.get_activation_snapshot(10);
    assert(snap.size() == 2);
    // Concept 1 should have higher activation (first in chain)
    bool found1 = false, found2 = false;
    for (auto& [cid, val] : snap) {
        if (cid == 1) { found1 = true; assert(val > 0.0); }
        if (cid == 2) { found2 = true; assert(val > 0.0); }
    }
    assert(found1 && found2);
    PASS();
}

void test_prune_activations() {
    TEST("Prune keeps max concepts");
    GDOConfig cfg;
    cfg.max_activated_concepts = 5;
    cfg.tick_interval = std::chrono::milliseconds(20);
    cfg.decay_rate = 0.001;  // minimal decay
    cfg.enable_autonomous_thinking = false;
    GlobalDynamicsOperator gdo(cfg);

    // Inject 10 concepts
    for (ConceptId c = 1; c <= 10; ++c) {
        gdo.inject_seeds({c}, static_cast<double>(c) / 10.0);
    }

    gdo.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    gdo.stop();

    auto snap = gdo.get_activation_snapshot(20);
    assert(snap.size() <= 5);
    PASS();
}

void test_double_start_stop() {
    TEST("Double start/stop is safe");
    GlobalDynamicsOperator gdo;
    gdo.start();
    gdo.start();  // Should be no-op
    assert(gdo.is_running());
    gdo.stop();
    gdo.stop();  // Should be no-op
    assert(!gdo.is_running());
    PASS();
}

int main() {
    std::cout << "=== GlobalDynamicsOperator Tests ===" << std::endl;

    test_construct_default();
    test_inject_energy();
    test_inject_energy_cap();
    test_inject_seeds();
    test_start_stop();
    test_decay();
    test_snapshot();
    test_thinking_callback();
    test_no_thinking_below_threshold();
    test_feed_traversal();
    test_prune_activations();
    test_double_start_stop();

    std::cout << "\n=== " << tests_passed << "/" << tests_total << " PASSED ===" << std::endl;
    return (tests_passed == tests_total) ? 0 : 1;
}
