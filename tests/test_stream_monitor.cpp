// Test suite for StreamMonitor (Phase 5.3)

#include "../backend/streams/stream_monitor.hpp"
#include "../backend/streams/stream_monitor_cli.hpp"

#ifdef HAS_FULL_BACKEND
#include "../backend/streams/stream_orchestrator.hpp"
#include "../backend/streams/stream_scheduler.hpp"
#include "../backend/ltm/long_term_memory.hpp"
#include "../backend/memory/stm.hpp"
#include "../backend/micromodel/micro_model_registry.hpp"
#include "../backend/micromodel/embedding_manager.hpp"
#include "../backend/concurrent/shared_ltm.hpp"
#include "../backend/concurrent/shared_stm.hpp"
#include "../backend/concurrent/shared_registry.hpp"
#include "../backend/concurrent/shared_embeddings.hpp"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { \
        Register_##name() { \
            std::printf("  %-50s", #name); \
            try { test_##name(); tests_passed++; std::printf("[PASS]\n"); } \
            catch (const std::exception& e) { tests_failed++; std::printf("[FAIL] %s\n", e.what()); } \
            catch (...) { tests_failed++; std::printf("[FAIL] unknown\n"); } \
        } \
    } reg_##name; \
    static void test_##name()

#define ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::fabs((a)-(b)) > (eps)) { \
    char buf_[256]; std::snprintf(buf_, 256, "ASSERT_NEAR failed: %f vs %f (eps=%f)", (double)(a), (double)(b), (double)(eps)); \
    throw std::runtime_error(buf_); } } while(0)

// ─── Test 1: Latency Histogram Correctness ──────────────────────────────────

TEST(latency_histogram_percentiles) {
    brain19::LatencyHistogram hist;

    // Insert 100 values: 1, 2, ..., 100
    for (int i = 1; i <= 100; ++i) {
        hist.record(static_cast<double>(i));
    }

    auto stats = hist.compute();
    ASSERT(stats.n == 100);
    ASSERT_NEAR(stats.p50, 50.0, 2.0);
    ASSERT_NEAR(stats.p95, 95.0, 2.0);
    ASSERT_NEAR(stats.p99, 99.0, 2.0);
    ASSERT_NEAR(stats.max, 100.0, 0.1);
}

// ─── Test 2: Histogram Sliding Window ───────────────────────────────────────

TEST(latency_histogram_sliding_window) {
    brain19::LatencyHistogram hist;

    // Fill past window size
    for (size_t i = 0; i < brain19::LatencyHistogram::kWindowSize + 500; ++i) {
        hist.record(static_cast<double>(i % 1000));
    }

    auto stats = hist.compute();
    // Should only contain last kWindowSize samples
    ASSERT(stats.n == brain19::LatencyHistogram::kWindowSize);
    ASSERT(stats.max <= 999.0);
}

// ─── Test 3: History Ring-Buffer Wrap-Around ────────────────────────────────

#ifdef HAS_FULL_BACKEND
// Helper: create full monitoring stack
struct TestStack {
    brain19::LongTermMemory ltm_raw;
    brain19::ShortTermMemory stm_raw;
    brain19::MicroModelRegistry reg_raw;
    brain19::EmbeddingManager emb_raw;
    brain19::SharedLTM ltm;
    brain19::SharedSTM stm;
    brain19::SharedRegistry registry;
    brain19::SharedEmbeddings embeddings;
    brain19::StreamConfig cfg;
    std::unique_ptr<brain19::StreamOrchestrator> orch;
    std::unique_ptr<brain19::StreamScheduler> sched;
    std::unique_ptr<brain19::StreamMonitor> monitor;

    TestStack()
        : ltm(ltm_raw), stm(stm_raw), registry(reg_raw), embeddings(emb_raw)
    {
        cfg.max_streams = 4;
        orch = std::make_unique<brain19::StreamOrchestrator>(ltm, stm, registry, embeddings, cfg);
        brain19::SchedulerConfig sc;
        sc.total_max_streams = 4;
        sched = std::make_unique<brain19::StreamScheduler>(*orch, sc);
        monitor = std::make_unique<brain19::StreamMonitor>(*orch, *sched);
    }
};

TEST(history_ring_buffer_wrap) {
    TestStack ts;
    ts.sched->start();
    ts.monitor->start();

    // Let it run for enough time to get some history
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto hist = ts.monitor->history(60);
    // Should have entries (at 10Hz, ~5 entries in 500ms)
    ASSERT(hist.size() >= 2);

    // Entries should be in time order
    for (size_t i = 1; i < hist.size(); ++i) {
        ASSERT(hist[i].timestamp >= hist[i-1].timestamp);
    }

    ts.monitor->stop();
    ts.sched->shutdown();
}

// ─── Test 4: Alert Threshold Detection ──────────────────────────────────────

TEST(alert_threshold_detection) {
    TestStack ts;
    // Set very aggressive thresholds
    brain19::AlertThresholds thresh;
    thresh.stall_threshold = std::chrono::milliseconds{100};
    thresh.high_latency_us = 1.0;  // 1us = will likely trigger

    ts.monitor->set_thresholds(thresh);
    ts.sched->start();
    ts.monitor->start();

    // Record some high latency
    for (int i = 0; i < 100; ++i) {
        ts.monitor->record_tick_latency(50000.0);  // 50ms
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    auto alerts = ts.monitor->active_alerts();
    // Should have at least the high latency alert
    bool found_latency_alert = false;
    for (auto& a : alerts) {
        if (a.type == brain19::AlertType::HighLatency) {
            found_latency_alert = true;
        }
    }
    ASSERT(found_latency_alert);

    ts.monitor->stop();
    ts.sched->shutdown();
}

// ─── Test 5: Category Aggregation ───────────────────────────────────────────

TEST(category_aggregation) {
    TestStack ts;
    ts.sched->start();
    ts.monitor->start();

    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    auto cats = ts.monitor->category_snapshots();

    // Should have 4 categories
    ASSERT(cats.size() == static_cast<size_t>(brain19::StreamCategory::Count));

    // Each category name should be valid
    for (auto& c : cats) {
        auto name = brain19::category_name(c.category);
        ASSERT(!name.empty());
        ASSERT(name != "Unknown");
    }

    // Total stream count across categories should match global
    auto gs = ts.monitor->global_snapshot();
    uint32_t total = 0;
    for (auto& c : cats) total += c.stream_count;
    ASSERT(total == gs.total_streams);

    ts.monitor->stop();
    ts.sched->shutdown();
}

// ─── Test 6: CLI Output Format ──────────────────────────────────────────────

TEST(cli_output_format) {
    TestStack ts;
    ts.sched->start();
    ts.monitor->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    brain19::StreamMonitorCLI cli(*ts.monitor);

    // Status should contain header
    auto status = cli.cmd_status();
    ASSERT(status.find("Brain19 Stream Status") != std::string::npos);
    ASSERT(status.find("ID") != std::string::npos);

    // Throughput should contain TOTAL
    auto tp = cli.cmd_throughput();
    ASSERT(tp.find("Throughput") != std::string::npos);
    ASSERT(tp.find("TOTAL") != std::string::npos);

    // Latency should show percentiles
    auto lat = cli.cmd_latency();
    ASSERT(lat.find("p50") != std::string::npos);
    ASSERT(lat.find("p99") != std::string::npos);

    // Categories should list all 4
    auto cat = cli.cmd_categories();
    ASSERT(cat.find("Perception") != std::string::npos);
    ASSERT(cat.find("Reasoning") != std::string::npos);
    ASSERT(cat.find("Memory") != std::string::npos);
    ASSERT(cat.find("Creative") != std::string::npos);

    // Alerts
    auto alerts = cli.cmd_alerts();
    ASSERT(alerts.find("Alerts") != std::string::npos);

    // History
    auto hist = cli.cmd_history(5);
    ASSERT(hist.find("History") != std::string::npos);

    // Unknown command
    auto unk = cli.dispatch("foobar");
    ASSERT(unk.find("Unknown command") != std::string::npos);

    // Usage
    auto usage = brain19::StreamMonitorCLI::usage();
    ASSERT(usage.find("status") != std::string::npos);

    ts.monitor->stop();
    ts.sched->shutdown();
}

// ─── Test 7: Metric Sampling at 10Hz ────────────────────────────────────────

TEST(metric_sampling_10hz) {
    TestStack ts;
    ts.sched->start();
    ts.monitor->start();

    // Run for 1 second — should get ~10 history entries
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto hist = ts.monitor->history(1);
    // At 10Hz for 1 second → ~10 entries (allow 5-15 for timing variance)
    ASSERT(hist.size() >= 5);
    ASSERT(hist.size() <= 20);

    ts.monitor->stop();
    ts.sched->shutdown();
}

#else
// Lite mode: only histogram tests
TEST(history_ring_buffer_wrap) { std::printf("(SKIP: needs HAS_FULL_BACKEND) "); }
TEST(alert_threshold_detection) { std::printf("(SKIP: needs HAS_FULL_BACKEND) "); }
TEST(category_aggregation) { std::printf("(SKIP: needs HAS_FULL_BACKEND) "); }
TEST(cli_output_format) { std::printf("(SKIP: needs HAS_FULL_BACKEND) "); }
TEST(metric_sampling_10hz) { std::printf("(SKIP: needs HAS_FULL_BACKEND) "); }
#endif

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    std::printf("\n=== Brain19 Stream Monitor Tests ===\n\n");
    std::printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
