#pragma once

#include "think_stream.hpp"
#include "stream_categories.hpp"
#include "stream_orchestrator.hpp"
#include "stream_scheduler.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace brain19 {

// ─── Latency Histogram (mutex-guarded sliding window) ────────────────────────

class LatencyHistogram {
public:
    static constexpr size_t kWindowSize = 1000;

    void record(double latency_us);
    void reset();

    size_t count() const { return count_.load(std::memory_order_relaxed); }

    // Returns sorted snapshot for percentile queries
    struct Stats {
        double p50 = 0, p95 = 0, p99 = 0, max = 0;
        size_t n = 0;
    };
    Stats compute() const;

private:
    mutable std::mutex mtx_;
    std::array<double, kWindowSize> samples_{};
    std::atomic<size_t> count_{0};
    size_t write_pos_ = 0;
};

// ─── Alert ──────────────────────────────────────────────────────────────────

enum class AlertLevel : uint8_t { Warning, Critical };
enum class AlertType : uint8_t { Stall, ThroughputDrop, HighLatency, ErrorRate };

struct Alert {
    AlertLevel level;
    AlertType type;
    uint32_t stream_id;  // 0 = global
    std::string message;
    std::chrono::steady_clock::time_point timestamp;
};

struct AlertThresholds {
    std::chrono::milliseconds stall_threshold{5000};
    double throughput_drop_pct = 50.0;       // % drop from baseline triggers alert
    double high_latency_us = 10000.0;        // p99 above this triggers alert
    double error_rate_threshold = 0.01;       // >1% errors
};

// ─── Per-Stream Snapshot ────────────────────────────────────────────────────

struct StreamSnapshot {
    uint32_t stream_id = 0;
    StreamCategory category = StreamCategory::Perception;
    StreamState state = StreamState::Created;
    double ticks_per_sec = 0.0;
    double idle_pct = 0.0;
    double avg_tick_latency_us = 0.0;
    size_t queue_depth = 0;
    uint64_t total_ticks = 0;
    uint64_t errors = 0;
};

// ─── Per-Category Snapshot ──────────────────────────────────────────────────

struct CategorySnapshot {
    StreamCategory category;
    uint32_t stream_count = 0;
    double total_throughput = 0.0;   // ticks/sec aggregated
    double avg_idle_pct = 0.0;
    uint64_t total_ticks = 0;
};

// ─── Global Snapshot ────────────────────────────────────────────────────────

struct GlobalSnapshot {
    double total_throughput = 0.0;
    uint32_t active_streams = 0;
    uint32_t total_streams = 0;
    double system_load = 0.0;  // 0..1 estimated utilization
    std::chrono::steady_clock::time_point timestamp;
};

// ─── History Entry ──────────────────────────────────────────────────────────

struct HistoryEntry {
    double total_throughput = 0.0;
    uint32_t active_streams = 0;
    double p99_latency_us = 0.0;
    std::chrono::steady_clock::time_point timestamp;
};

// ─── Stream Monitor ─────────────────────────────────────────────────────────

class StreamMonitor {
public:
    static constexpr size_t kHistoryCapacity = 600;  // 60s at 10Hz
    static constexpr auto kSampleInterval = std::chrono::milliseconds{100}; // 10Hz

    StreamMonitor(StreamOrchestrator& orchestrator, StreamScheduler& scheduler,
                  AlertThresholds thresholds = {});
    ~StreamMonitor();

    StreamMonitor(const StreamMonitor&) = delete;
    StreamMonitor& operator=(const StreamMonitor&) = delete;

    // Lifecycle
    void start();
    void stop();
    bool is_running() const { return running_.load(std::memory_order_acquire); }

    // Snapshots (thread-safe, returns copies)
    std::vector<StreamSnapshot> stream_snapshots() const;
    std::array<CategorySnapshot, static_cast<size_t>(StreamCategory::Count)> category_snapshots() const;
    GlobalSnapshot global_snapshot() const;

    // Latency
    LatencyHistogram::Stats latency_stats() const;
    void record_tick_latency(double latency_us);

    // Alerts
    std::vector<Alert> active_alerts() const;
    void set_thresholds(AlertThresholds t);

    // History
    std::vector<HistoryEntry> history(size_t last_n_seconds = 60) const;

private:
    void sample_loop();
    void sample_once();
    void check_alerts();
    void push_history(const GlobalSnapshot& gs);

    StreamOrchestrator& orchestrator_;
    StreamScheduler& scheduler_;

    std::atomic<bool> running_{false};
    std::thread sample_thread_;

    // Cached snapshots
    mutable std::mutex snap_mtx_;
    std::vector<StreamSnapshot> cached_streams_;
    std::array<CategorySnapshot, static_cast<size_t>(StreamCategory::Count)> cached_categories_{};
    GlobalSnapshot cached_global_;

    // Previous tick counts for rate computation
    struct PrevStreamData {
        uint32_t stream_id = 0;
        uint64_t ticks = 0;
        uint64_t idle_ticks = 0;
        std::chrono::steady_clock::time_point time;
    };
    std::vector<PrevStreamData> prev_data_;
    std::chrono::steady_clock::time_point prev_sample_time_;

    // Latency histogram
    LatencyHistogram latency_hist_;

    // Alerts
    mutable std::mutex alert_mtx_;
    AlertThresholds thresholds_;
    std::vector<Alert> alerts_;
    double baseline_throughput_ = 0.0;

    // History ring buffer
    mutable std::mutex hist_mtx_;
    std::array<HistoryEntry, kHistoryCapacity> history_buf_{};
    size_t history_write_ = 0;
    size_t history_count_ = 0;
};

} // namespace brain19
