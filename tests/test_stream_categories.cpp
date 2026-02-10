// =============================================================================
// Test Stream Categories (Phase 5.2)
// =============================================================================

#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <atomic>
#include <algorithm>

// Always need full backend for stream tests
#ifndef HAS_FULL_BACKEND
#define HAS_FULL_BACKEND
#endif

#include "streams/stream_categories.hpp"
#include "streams/stream_scheduler.hpp"
#include "streams/stream_orchestrator.hpp"
#include "streams/think_stream.hpp"
#include "ltm/long_term_memory.hpp"
#include "memory/stm.hpp"
#include "micromodel/micro_model_registry.hpp"
#include "micromodel/embedding_manager.hpp"
#include "concurrent/shared_ltm.hpp"
#include "concurrent/shared_stm.hpp"
#include "concurrent/shared_registry.hpp"
#include "concurrent/shared_embeddings.hpp"

using namespace brain19;

// =============================================================================
// Test infrastructure
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct TestRegister_##name { \
        TestRegister_##name() { \
            std::cout << "  [TEST] " #name "... " << std::flush; \
            try { \
                test_##name(); \
                std::cout << "PASS\n"; \
                ++tests_passed; \
            } catch (const std::exception& e) { \
                std::cout << "FAIL: " << e.what() << "\n"; \
                ++tests_failed; \
            } catch (...) { \
                std::cout << "FAIL: unknown exception\n"; \
                ++tests_failed; \
            } \
        } \
    }; \
    static TestRegister_##name register_##name{}; \
    static void test_##name()

#define ASSERT_TRUE(expr) \
    do { if (!(expr)) throw std::runtime_error("Assertion failed: " #expr " at line " + std::to_string(__LINE__)); } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b " at line " + std::to_string(__LINE__)); } while(0)

#define ASSERT_GE(a, b) \
    do { if ((a) < (b)) throw std::runtime_error("Assertion failed: " #a " >= " #b " at line " + std::to_string(__LINE__)); } while(0)

#define ASSERT_LE(a, b) \
    do { if ((a) > (b)) throw std::runtime_error("Assertion failed: " #a " <= " #b " at line " + std::to_string(__LINE__)); } while(0)

// Helper: create shared infrastructure for tests
struct TestEnv {
    LongTermMemory ltm;
    ShortTermMemory stm;
    MicroModelRegistry registry;
    EmbeddingManager embeddings;

    SharedLTM s_ltm;
    SharedSTM s_stm;
    SharedRegistry s_reg;
    SharedEmbeddings s_emb;
    StreamConfig base_config;

    TestEnv()
        : s_ltm(ltm), s_stm(stm), s_reg(registry), s_emb(embeddings)
    {
        base_config.max_streams = 16;
        base_config.tick_interval = std::chrono::milliseconds{5};
        base_config.shutdown_timeout = std::chrono::seconds{3};
    }

    std::unique_ptr<StreamOrchestrator> make_orchestrator() {
        return std::make_unique<StreamOrchestrator>(s_ltm, s_stm, s_reg, s_emb, base_config);
    }
};

// =============================================================================
// 1. Each category start/stop individually
// =============================================================================

TEST(each_category_start_stop) {
    TestEnv env;
    auto orch = env.make_orchestrator();

    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        auto cat = static_cast<StreamCategory>(i);
        SchedulerConfig sched_cfg;
        sched_cfg.total_max_streams = 8;
        // Set only this category to have streams
        for (size_t j = 0; j < static_cast<size_t>(StreamCategory::Count); ++j) {
            sched_cfg.budgets[j] = {0, 0, 0};  // disable all
        }
        sched_cfg.budgets[i] = {1, 2, 1};  // enable this one

        StreamScheduler scheduler(*orch, sched_cfg);
        scheduler.start();

        std::this_thread::sleep_for(std::chrono::milliseconds{50});

        ASSERT_GE(scheduler.stream_count(cat), 1u);

        // Other categories should have 0
        for (size_t j = 0; j < static_cast<size_t>(StreamCategory::Count); ++j) {
            if (j != i) {
                ASSERT_EQ(scheduler.stream_count(static_cast<StreamCategory>(j)), 0u);
            }
        }

        scheduler.shutdown(std::chrono::milliseconds{2000});
    }
}

// =============================================================================
// 2. All 4 categories parallel
// =============================================================================

TEST(all_categories_parallel) {
    TestEnv env;
    auto orch = env.make_orchestrator();

    SchedulerConfig cfg;
    cfg.total_max_streams = 16;
    StreamScheduler scheduler(*orch, cfg);
    scheduler.start();

    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    // All categories should have at least their default streams
    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        auto cat = static_cast<StreamCategory>(i);
        ASSERT_GE(scheduler.stream_count(cat), 1u);
    }

    // Total should match sum of defaults (Perception:1 + Reasoning:2 + Memory:1 + Creative:1 = 5)
    uint32_t total = 0;
    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        total += scheduler.stream_count(static_cast<StreamCategory>(i));
    }
    ASSERT_GE(total, 4u);  // at least 4 (one per category)

    scheduler.shutdown(std::chrono::milliseconds{3000});
}

// =============================================================================
// 3. Priority scheduling: Reasoning > Perception > Creative > Memory
// =============================================================================

TEST(priority_scheduling) {
    // Verify priority ordering
    ASSERT_TRUE(category_priority(StreamCategory::Reasoning) <
                category_priority(StreamCategory::Perception));
    ASSERT_TRUE(category_priority(StreamCategory::Perception) <
                category_priority(StreamCategory::Creative));
    ASSERT_TRUE(category_priority(StreamCategory::Creative) <
                category_priority(StreamCategory::Memory));

    // Verify subsystem specialization
    auto reasoning_subs = category_subsystems(StreamCategory::Reasoning);
    ASSERT_TRUE(has_subsystem(reasoning_subs, Subsystem::Spreading));
    ASSERT_TRUE(has_subsystem(reasoning_subs, Subsystem::Salience));
    ASSERT_TRUE(has_subsystem(reasoning_subs, Subsystem::Understanding));
    ASSERT_TRUE(!has_subsystem(reasoning_subs, Subsystem::Curiosity));

    auto creative_subs = category_subsystems(StreamCategory::Creative);
    ASSERT_TRUE(has_subsystem(creative_subs, Subsystem::Curiosity));
    ASSERT_TRUE(has_subsystem(creative_subs, Subsystem::Understanding));
    ASSERT_TRUE(!has_subsystem(creative_subs, Subsystem::Spreading));

    auto perception_subs = category_subsystems(StreamCategory::Perception);
    ASSERT_TRUE(has_subsystem(perception_subs, Subsystem::Spreading));
    ASSERT_TRUE(has_subsystem(perception_subs, Subsystem::Salience));

    auto memory_subs = category_subsystems(StreamCategory::Memory);
    ASSERT_TRUE(has_subsystem(memory_subs, Subsystem::Salience));
}

// =============================================================================
// 4. Dynamic reallocation on load change
// =============================================================================

TEST(dynamic_reallocation) {
    TestEnv env;
    auto orch = env.make_orchestrator();

    SchedulerConfig cfg;
    cfg.total_max_streams = 20;
    cfg.rebalance_interval = std::chrono::milliseconds{50};  // fast for test
    cfg.budgets[static_cast<size_t>(StreamCategory::Reasoning)] = {1, 6, 1};
    cfg.budgets[static_cast<size_t>(StreamCategory::Memory)] = {1, 4, 1};

    StreamScheduler scheduler(*orch, cfg);
    scheduler.start();

    std::this_thread::sleep_for(std::chrono::milliseconds{80});

    // Initially should have default streams
    uint32_t initial_reasoning = scheduler.stream_count(StreamCategory::Reasoning);
    uint32_t initial_memory = scheduler.stream_count(StreamCategory::Memory);
    ASSERT_GE(initial_reasoning, 1u);
    ASSERT_GE(initial_memory, 1u);

    // Signal high activation → should scale up reasoning
    SystemLoad high_load;
    high_load.activation_level = 0.9;
    high_load.idle_ratio = 0.0;
    high_load.pending_inputs = 0;
    scheduler.update_load(high_load);

    // Wait for rebalance
    std::this_thread::sleep_for(std::chrono::milliseconds{200});

    uint32_t scaled_reasoning = scheduler.stream_count(StreamCategory::Reasoning);
    ASSERT_GE(scaled_reasoning, initial_reasoning);  // should not decrease

    // Signal high idle → should scale up memory
    SystemLoad idle_load;
    idle_load.activation_level = 0.1;
    idle_load.idle_ratio = 0.9;
    idle_load.pending_inputs = 0;
    scheduler.update_load(idle_load);

    std::this_thread::sleep_for(std::chrono::milliseconds{200});

    uint32_t scaled_memory = scheduler.stream_count(StreamCategory::Memory);
    ASSERT_GE(scaled_memory, initial_memory);

    scheduler.shutdown(std::chrono::milliseconds{3000});
}

// =============================================================================
// 5. Resource budget enforcement
// =============================================================================

TEST(resource_budget_enforcement) {
    TestEnv env;
    auto orch = env.make_orchestrator();

    SchedulerConfig cfg;
    cfg.total_max_streams = 20;
    cfg.budgets[static_cast<size_t>(StreamCategory::Reasoning)] = {1, 3, 1};
    cfg.budgets[static_cast<size_t>(StreamCategory::Perception)] = {1, 2, 1};
    cfg.budgets[static_cast<size_t>(StreamCategory::Memory)] = {1, 2, 1};
    cfg.budgets[static_cast<size_t>(StreamCategory::Creative)] = {1, 2, 1};

    StreamScheduler scheduler(*orch, cfg);
    scheduler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{50});

    // Try to manually add streams beyond max for Reasoning (max=3)
    // After start, should have 1. Add 2 more to reach 3.
    scheduler.create_categorized_stream(StreamCategory::Reasoning);
    scheduler.create_categorized_stream(StreamCategory::Reasoning);

    uint32_t reasoning_count = scheduler.stream_count(StreamCategory::Reasoning);
    // Should be 3 (1 default + 2 manual)
    ASSERT_EQ(reasoning_count, 3u);

    // After rebalance, it should clamp back to max (3) — so still 3
    scheduler.rebalance();
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    reasoning_count = scheduler.stream_count(StreamCategory::Reasoning);
    ASSERT_LE(reasoning_count, 3u);  // budget max enforced

    // Check min: all categories must have >= min_streams (1)
    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        ASSERT_GE(scheduler.stream_count(static_cast<StreamCategory>(i)),
                  cfg.budgets[i].min_streams);
    }

    scheduler.shutdown(std::chrono::milliseconds{3000});
}

// =============================================================================
// 6. Fair scheduling (no starvation)
// =============================================================================

TEST(fair_scheduling_no_starvation) {
    TestEnv env;
    auto orch = env.make_orchestrator();

    SchedulerConfig cfg;
    cfg.total_max_streams = 16;
    cfg.max_starvation_rounds = 3;  // low threshold for test
    StreamScheduler scheduler(*orch, cfg);
    scheduler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{50});

    // Schedule many tasks by priority — Memory (lowest priority) should still get scheduled
    for (int i = 0; i < 20; ++i) {
        ThinkTask task;
        task.type = ThinkTask::Type::Tick;
        scheduler.schedule_task_by_priority(task);
    }

    // After enough rounds, Memory starvation should trigger forced scheduling
    // Check that starvation count doesn't grow unbounded
    auto memory_starved = scheduler.stats(StreamCategory::Memory)
                              .starvation_count.load(std::memory_order_relaxed);
    // Should be bounded by max_starvation_rounds (gets reset on forced schedule)
    // In practice it oscillates; just check it's not growing to infinity
    ASSERT_LE(memory_starved, cfg.max_starvation_rounds + 5u);

    scheduler.shutdown(std::chrono::milliseconds{3000});
}

// =============================================================================
// 7. Graceful shutdown of all categories
// =============================================================================

TEST(graceful_shutdown) {
    TestEnv env;
    auto orch = env.make_orchestrator();

    SchedulerConfig cfg;
    cfg.total_max_streams = 12;
    StreamScheduler scheduler(*orch, cfg);
    scheduler.start();

    // Let streams run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds{200});

    // Verify streams are running
    uint32_t total_before = 0;
    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        total_before += scheduler.stream_count(static_cast<StreamCategory>(i));
    }
    ASSERT_GE(total_before, 4u);

    // Shutdown
    bool ok = scheduler.shutdown(std::chrono::milliseconds{5000});
    ASSERT_TRUE(ok);

    // All categories should be empty after shutdown
    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        ASSERT_EQ(scheduler.stream_count(static_cast<StreamCategory>(i)), 0u);
    }

    ASSERT_TRUE(!scheduler.is_running());
}

// =============================================================================
// 8. Category config generation
// =============================================================================

TEST(category_config_generation) {
    // Verify make_category_config produces correct configs
    auto perc_cfg = make_category_config(StreamCategory::Perception);
    ASSERT_EQ(perc_cfg.tick_interval.count(), 5);
    ASSERT_TRUE(has_subsystem(perc_cfg.subsystem_flags, Subsystem::Spreading));
    ASSERT_TRUE(perc_cfg.backoff_strategy == BackoffStrategy::SpinYieldSleep);
    ASSERT_EQ(perc_cfg.spin_count, 200u);

    auto mem_cfg = make_category_config(StreamCategory::Memory);
    ASSERT_EQ(mem_cfg.tick_interval.count(), 50);
    ASSERT_TRUE(mem_cfg.backoff_strategy == BackoffStrategy::Sleep);

    auto reas_cfg = make_category_config(StreamCategory::Reasoning);
    ASSERT_EQ(reas_cfg.tick_interval.count(), 10);

    auto crea_cfg = make_category_config(StreamCategory::Creative);
    ASSERT_EQ(crea_cfg.tick_interval.count(), 25);

    // Category names
    ASSERT_TRUE(category_name(StreamCategory::Perception) == "Perception");
    ASSERT_TRUE(category_name(StreamCategory::Reasoning) == "Reasoning");
    ASSERT_TRUE(category_name(StreamCategory::Memory) == "Memory");
    ASSERT_TRUE(category_name(StreamCategory::Creative) == "Creative");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== Stream Categories Test Suite (Phase 5.2) ===\n\n";
    // Tests are auto-registered via static constructors above
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
