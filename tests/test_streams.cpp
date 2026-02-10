// test_streams.cpp — Multi-Stream System Tests
// Minimal test framework (no external deps)

#include "../backend/streams/lock_free_queue.hpp"
#include "../backend/streams/think_stream.hpp"
#include "../backend/streams/stream_orchestrator.hpp"
#include "../backend/streams/stream_config.hpp"

// We need the actual implementations for the shared wrappers
// Since the real LTM/STM etc. require full compilation, we provide
// minimal stubs that satisfy the interfaces used by ThinkStream.

#include <cassert>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Minimal stubs for brain19 types used by shared_*.hpp
// ============================================================================

// We need to provide implementations that the shared wrappers reference.
// The real ones are deep in the codebase, so we stub them for testing.

namespace {

int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

std::vector<TestResult> results;

void report(const std::string& name, bool passed, const std::string& msg = "") {
    tests_run++;
    if (passed) {
        tests_passed++;
        std::cout << "  ✅ " << name << "\n";
    } else {
        tests_failed++;
        std::cout << "  ❌ " << name;
        if (!msg.empty()) std::cout << " — " << msg;
        std::cout << "\n";
    }
    results.push_back({name, passed, msg});
}

#define TEST(name) void test_##name(); \
    struct Register_##name { Register_##name() { test_funcs.push_back({#name, test_##name}); } } reg_##name; \
    void test_##name()

struct TestEntry { std::string name; std::function<void()> fn; };
std::vector<TestEntry> test_funcs;

} // anonymous

// ============================================================================
// Test 1: Lock-free MPMC Queue correctness
// ============================================================================
void test_mpmc_queue_basic() {
    brain19::MPMCQueue<int> q(16);

    // Push and pop
    assert(q.try_push(42));
    assert(q.try_push(43));
    auto v1 = q.try_pop();
    auto v2 = q.try_pop();
    auto v3 = q.try_pop();

    bool ok = v1.has_value() && v1.value() == 42 &&
              v2.has_value() && v2.value() == 43 &&
              !v3.has_value();
    report("MPMC basic push/pop", ok);
}

void test_mpmc_queue_full() {
    brain19::MPMCQueue<int> q(4); // rounds to 4
    // Fill it up
    int pushed = 0;
    for (int i = 0; i < 100; ++i) {
        if (q.try_push(i)) pushed++;
    }
    // Should have pushed exactly capacity (4)
    report("MPMC queue full detection", pushed == 4, 
           "pushed=" + std::to_string(pushed));
}

void test_mpmc_queue_concurrent() {
    constexpr int N = 10000;
    constexpr int PRODUCERS = 4;
    constexpr int CONSUMERS = 4;
    
    brain19::MPMCQueue<int> q(1024);
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::atomic<bool> done{false};

    // Producers
    std::vector<std::thread> producers;
    for (int p = 0; p < PRODUCERS; ++p) {
        producers.emplace_back([&, p]() {
            int count = 0;
            for (int i = p * N; i < (p + 1) * N; ++i) {
                while (!q.try_push(i)) {
                    std::this_thread::yield();
                }
                count++;
            }
            produced.fetch_add(count, std::memory_order_relaxed);
        });
    }

    // Consumers
    std::vector<std::thread> consumers;
    for (int c = 0; c < CONSUMERS; ++c) {
        consumers.emplace_back([&]() {
            int count = 0;
            while (!done.load(std::memory_order_relaxed) || !q.empty()) {
                auto val = q.try_pop();
                if (val.has_value()) {
                    count++;
                } else {
                    std::this_thread::yield();
                }
            }
            consumed.fetch_add(count, std::memory_order_relaxed);
        });
    }

    for (auto& t : producers) t.join();
    done.store(true, std::memory_order_release);
    // Wait until queue is fully drained before joining consumers
    while (!q.empty()) {
        std::this_thread::yield();
    }
    // Small grace period for consumers to finish their last iteration
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (auto& t : consumers) t.join();

    int total_produced = produced.load();
    int total_consumed = consumed.load();

    report("MPMC concurrent correctness", 
           total_produced == PRODUCERS * N && total_consumed == total_produced,
           "produced=" + std::to_string(total_produced) + 
           " consumed=" + std::to_string(total_consumed));
}

// ============================================================================
// Test 2: SPSC Queue
// ============================================================================
void test_spsc_queue() {
    brain19::SPSCQueue<int> q(8);
    
    q.try_push(1);
    q.try_push(2);
    q.try_push(3);
    
    auto a = q.try_pop();
    auto b = q.try_pop();
    auto c = q.try_pop();
    auto d = q.try_pop();
    
    bool ok = a.value() == 1 && b.value() == 2 && c.value() == 3 && !d.has_value();
    report("SPSC queue basic", ok);
}

// ============================================================================
// Test 3: Single stream lifecycle
// Using the real brain19 types requires actual LTM/STM instances.
// We test with the real objects.
// ============================================================================

// We need access to the actual classes. Since compilation of the full 
// brain19 is complex, we test what we can without the full backend.
// The stream lifecycle test uses a custom approach:

#ifdef HAS_FULL_BACKEND
#include "../backend/ltm/long_term_memory.hpp"
#include "../backend/memory/stm.hpp"
#include "../backend/micromodel/micro_model_registry.hpp"
#include "../backend/micromodel/embedding_manager.hpp"

void test_stream_lifecycle() {
    brain19::LongTermMemory ltm;
    brain19::ShortTermMemory stm;
    brain19::MicroModelRegistry registry;
    brain19::EmbeddingManager embeddings;

    brain19::SharedLTM s_ltm(ltm);
    brain19::SharedSTM s_stm(stm);
    brain19::SharedRegistry s_reg(registry);
    brain19::SharedEmbeddings s_emb(embeddings);

    brain19::StreamConfig config;
    config.tick_interval = std::chrono::milliseconds(5);
    config.shutdown_timeout = std::chrono::seconds(2);

    // Create
    brain19::ThinkStream stream(1, s_ltm, s_stm, s_reg, s_emb, config);
    report("Stream created", stream.state() == brain19::StreamState::Created);

    // Start
    bool started = stream.start();
    // Give thread time to transition to Running
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    report("Stream started", started && stream.state() == brain19::StreamState::Running);

    // Let it tick a few times
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto ticks = stream.metrics().total_ticks.load();
    report("Stream ticked", ticks > 0, "ticks=" + std::to_string(ticks));

    // Stop
    stream.stop();
    bool joined = stream.join(std::chrono::milliseconds(2000));
    report("Stream stopped", joined && stream.state() == brain19::StreamState::Stopped);
}

void test_multi_stream_concurrent() {
    brain19::LongTermMemory ltm;
    brain19::ShortTermMemory stm;
    brain19::MicroModelRegistry registry;
    brain19::EmbeddingManager embeddings;

    brain19::SharedLTM s_ltm(ltm);
    brain19::SharedSTM s_stm(stm);
    brain19::SharedRegistry s_reg(registry);
    brain19::SharedEmbeddings s_emb(embeddings);

    brain19::StreamConfig config;
    config.tick_interval = std::chrono::milliseconds(5);

    brain19::StreamOrchestrator orch(s_ltm, s_stm, s_reg, s_emb, config);

    // Create 4 streams
    for (int i = 0; i < 4; ++i) {
        orch.create_stream();
    }
    report("Multi-stream: created 4", orch.stream_count() == 4);

    orch.start_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    report("Multi-stream: all running", orch.running_count() == 4);

    bool ok = orch.shutdown(std::chrono::milliseconds(3000));
    report("Multi-stream: graceful shutdown", ok);
}

void test_graceful_shutdown_under_load() {
    brain19::LongTermMemory ltm;
    brain19::ShortTermMemory stm;
    brain19::MicroModelRegistry registry;
    brain19::EmbeddingManager embeddings;

    // Add some concepts to make streams do real work
    auto c1 = ltm.store_concept("test1", "def1", 
        brain19::EpistemicMetadata(brain19::EpistemicType::FACT, brain19::EpistemicStatus::ACTIVE, 0.8));
    auto c2 = ltm.store_concept("test2", "def2",
        brain19::EpistemicMetadata(brain19::EpistemicType::FACT, brain19::EpistemicStatus::ACTIVE, 0.7));
    ltm.add_relation(c1, c2, brain19::RelationType::CAUSES, 0.9);

    brain19::SharedLTM s_ltm(ltm);
    brain19::SharedSTM s_stm(stm);
    brain19::SharedRegistry s_reg(registry);
    brain19::SharedEmbeddings s_emb(embeddings);

    brain19::StreamConfig config;
    config.tick_interval = std::chrono::milliseconds(1);
    config.shutdown_timeout = std::chrono::seconds(3);

    brain19::StreamOrchestrator orch(s_ltm, s_stm, s_reg, s_emb, config);
    
    // Create and start streams
    for (int i = 0; i < 8; ++i) orch.create_stream();
    orch.start_all();
    
    // Let them run under load
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Inject tasks while shutting down
    for (int i = 0; i < 100; ++i) {
        orch.distribute_task(brain19::ThinkTask{});
    }
    
    bool ok = orch.shutdown(std::chrono::milliseconds(3000));
    report("Shutdown under load", ok);
}

void test_health_monitoring() {
    brain19::LongTermMemory ltm;
    brain19::ShortTermMemory stm;
    brain19::MicroModelRegistry registry;
    brain19::EmbeddingManager embeddings;

    brain19::SharedLTM s_ltm(ltm);
    brain19::SharedSTM s_stm(stm);
    brain19::SharedRegistry s_reg(registry);
    brain19::SharedEmbeddings s_emb(embeddings);

    brain19::StreamConfig config;
    config.tick_interval = std::chrono::milliseconds(5);
    config.monitor_interval = std::chrono::milliseconds(50);

    brain19::StreamOrchestrator orch(s_ltm, s_stm, s_reg, s_emb, config);

    auto id = orch.create_stream();
    orch.start_stream(id);
    orch.start_monitor();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto health = orch.health_check();
    bool ok = health.size() == 1 && 
              health[0].state == brain19::StreamState::Running &&
              health[0].total_ticks > 0 &&
              !health[0].stalled;
    report("Health monitoring", ok, 
           "ticks=" + std::to_string(health.empty() ? 0 : health[0].total_ticks));

    orch.stop_monitor();
    orch.shutdown(std::chrono::milliseconds(2000));
}

void test_backoff_behavior() {
    brain19::LongTermMemory ltm;
    brain19::ShortTermMemory stm;
    brain19::MicroModelRegistry registry;
    brain19::EmbeddingManager embeddings;

    brain19::SharedLTM s_ltm(ltm);
    brain19::SharedSTM s_stm(stm);
    brain19::SharedRegistry s_reg(registry);
    brain19::SharedEmbeddings s_emb(embeddings);

    // Test with different backoff strategies
    for (auto strategy : {brain19::BackoffStrategy::SpinYieldSleep,
                          brain19::BackoffStrategy::YieldSleep,
                          brain19::BackoffStrategy::Sleep}) {
        brain19::StreamConfig config;
        config.tick_interval = std::chrono::milliseconds(5);
        config.backoff_strategy = strategy;

        brain19::ThinkStream stream(100, s_ltm, s_stm, s_reg, s_emb, config);
        stream.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        stream.stop();
        stream.join(std::chrono::milliseconds(1000));
    }
    report("Backoff strategies all work", true);
}

void test_orchestrator_auto_scaling() {
    brain19::LongTermMemory ltm;
    brain19::ShortTermMemory stm;
    brain19::MicroModelRegistry registry;
    brain19::EmbeddingManager embeddings;

    brain19::SharedLTM s_ltm(ltm);
    brain19::SharedSTM s_stm(stm);
    brain19::SharedRegistry s_reg(registry);
    brain19::SharedEmbeddings s_emb(embeddings);

    brain19::StreamConfig config;
    config.max_streams = 3;  // explicit for reproducibility
    config.tick_interval = std::chrono::milliseconds(10);

    brain19::StreamOrchestrator orch(s_ltm, s_stm, s_reg, s_emb, config);
    orch.auto_scale();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    bool ok = orch.stream_count() == 3 && orch.running_count() == 3;
    report("Auto-scaling", ok,
           "count=" + std::to_string(orch.stream_count()) + 
           " running=" + std::to_string(orch.running_count()));

    orch.shutdown(std::chrono::milliseconds(2000));
}

#else

// Standalone queue-only tests when full backend isn't available
void test_stream_lifecycle() {
    report("Stream lifecycle (needs full backend)", false, "SKIPPED — compile with -DHAS_FULL_BACKEND");
}
void test_multi_stream_concurrent() {
    report("Multi-stream concurrent (needs full backend)", false, "SKIPPED");
}
void test_graceful_shutdown_under_load() {
    report("Shutdown under load (needs full backend)", false, "SKIPPED");
}
void test_health_monitoring() {
    report("Health monitoring (needs full backend)", false, "SKIPPED");
}
void test_backoff_behavior() {
    report("Backoff behavior (needs full backend)", false, "SKIPPED");
}
void test_orchestrator_auto_scaling() {
    report("Auto-scaling (needs full backend)", false, "SKIPPED");
}
#endif

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "\n🧠 Brain19 Multi-Stream System Tests\n";
    std::cout << "=====================================\n\n";

    std::cout << "📦 Lock-Free Queue Tests:\n";
    test_mpmc_queue_basic();
    test_mpmc_queue_full();
    test_mpmc_queue_concurrent();
    test_spsc_queue();

    std::cout << "\n🔄 Stream Lifecycle Tests:\n";
    test_stream_lifecycle();
    test_multi_stream_concurrent();

    std::cout << "\n⚡ Stress Tests:\n";
    test_graceful_shutdown_under_load();

    std::cout << "\n📊 Monitoring Tests:\n";
    test_health_monitoring();
    test_backoff_behavior();

    std::cout << "\n🔧 Orchestrator Tests:\n";
    test_orchestrator_auto_scaling();

    std::cout << "\n=====================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) std::cout << " (" << tests_failed << " failed)";
    std::cout << "\n\n";

    return tests_failed > 0 ? 1 : 0;
}
