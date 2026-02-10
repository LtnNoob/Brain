#include "stream_monitor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace brain19 {

// ─── LatencyHistogram ───────────────────────────────────────────────────────

void LatencyHistogram::record(double latency_us) {
    std::lock_guard lock(mtx_);
    samples_[write_pos_] = latency_us;
    write_pos_ = (write_pos_ + 1) % kWindowSize;
    count_.fetch_add(1, std::memory_order_relaxed);
}

void LatencyHistogram::reset() {
    std::lock_guard lock(mtx_);
    samples_.fill(0.0);
    write_pos_ = 0;
    count_.store(0, std::memory_order_relaxed);
}

LatencyHistogram::Stats LatencyHistogram::compute() const {
    std::lock_guard lock(mtx_);
    size_t n = std::min(count_.load(std::memory_order_relaxed), kWindowSize);
    if (n == 0) return {};

    std::vector<double> sorted(samples_.begin(), samples_.begin() + n);
    std::sort(sorted.begin(), sorted.end());

    Stats s;
    s.n = n;
    s.p50 = sorted[n * 50 / 100];
    s.p95 = sorted[std::min(n - 1, n * 95 / 100)];
    s.p99 = sorted[std::min(n - 1, n * 99 / 100)];
    s.max = sorted[n - 1];
    return s;
}

// ─── StreamMonitor ──────────────────────────────────────────────────────────

StreamMonitor::StreamMonitor(StreamOrchestrator& orchestrator,
                             StreamScheduler& scheduler,
                             AlertThresholds thresholds)
    : orchestrator_(orchestrator)
    , scheduler_(scheduler)
    , thresholds_(std::move(thresholds))
{
    prev_sample_time_ = std::chrono::steady_clock::now();
}

StreamMonitor::~StreamMonitor() {
    stop();
}

void StreamMonitor::start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) return;
    sample_thread_ = std::thread(&StreamMonitor::sample_loop, this);
}

void StreamMonitor::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    if (sample_thread_.joinable()) sample_thread_.join();
}

void StreamMonitor::sample_loop() {
    while (running_.load(std::memory_order_acquire)) {
        sample_once();
        check_alerts();
        std::this_thread::sleep_for(kSampleInterval);
    }
}

void StreamMonitor::sample_once() {
    auto now = std::chrono::steady_clock::now();
    auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(now - prev_sample_time_).count();
    double dt_sec = dt_us / 1e6;
    if (dt_sec < 0.001) dt_sec = 0.001;  // avoid div by zero

    auto health = orchestrator_.health_check();

    std::vector<StreamSnapshot> snaps;
    snaps.reserve(health.size());

    // Build prev_data lookup
    auto find_prev = [&](uint32_t id) -> PrevStreamData* {
        for (auto& p : prev_data_) {
            if (p.stream_id == id) return &p;
        }
        return nullptr;
    };

    std::vector<PrevStreamData> new_prev;
    new_prev.reserve(health.size());

    // Determine category for each stream
    auto get_category = [&](uint32_t id) -> StreamCategory {
        for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
            auto cat = static_cast<StreamCategory>(i);
            auto ids = scheduler_.get_streams(cat);
            for (auto sid : ids) {
                if (sid == id) return cat;
            }
        }
        return StreamCategory::Reasoning;  // default
    };

    for (auto& h : health) {
        StreamSnapshot ss;
        ss.stream_id = h.id;
        ss.state = h.state;
        ss.total_ticks = h.total_ticks;
        ss.errors = h.errors;
        ss.category = get_category(h.id);

        auto* prev = find_prev(h.id);
        if (prev) {
            uint64_t dticks = h.total_ticks - prev->ticks;
            ss.ticks_per_sec = dticks / dt_sec;

            // We don't have idle_ticks from StreamHealth directly,
            // approximate idle from state
            if (h.state == StreamState::Running && dticks == 0) {
                ss.idle_pct = 100.0;
            } else if (h.state != StreamState::Running) {
                ss.idle_pct = 100.0;
            } else {
                ss.idle_pct = 0.0;  // active
            }

            // tick latency approximation: dt / ticks if ticks > 0
            if (dticks > 0) {
                ss.avg_tick_latency_us = (dt_us) / static_cast<double>(dticks);
                record_tick_latency(ss.avg_tick_latency_us);
            }
        }
        ss.queue_depth = 0;  // StreamHealth doesn't expose this yet

        snaps.push_back(ss);
        new_prev.push_back({h.id, h.total_ticks, 0, now});
    }

    // Category aggregation
    std::array<CategorySnapshot, static_cast<size_t>(StreamCategory::Count)> cats{};
    for (size_t i = 0; i < cats.size(); ++i) {
        cats[i].category = static_cast<StreamCategory>(i);
    }
    for (auto& ss : snaps) {
        auto idx = static_cast<size_t>(ss.category);
        cats[idx].stream_count++;
        cats[idx].total_throughput += ss.ticks_per_sec;
        cats[idx].avg_idle_pct += ss.idle_pct;
        cats[idx].total_ticks += ss.total_ticks;
    }
    for (auto& c : cats) {
        if (c.stream_count > 0) {
            c.avg_idle_pct /= c.stream_count;
        }
    }

    // Global
    GlobalSnapshot gs;
    gs.timestamp = now;
    gs.total_streams = static_cast<uint32_t>(snaps.size());
    for (auto& ss : snaps) {
        gs.total_throughput += ss.ticks_per_sec;
        if (ss.state == StreamState::Running) gs.active_streams++;
    }
    if (gs.total_streams > 0) {
        gs.system_load = static_cast<double>(gs.active_streams) / gs.total_streams;
    }

    // Update baseline throughput (exponential moving average)
    if (baseline_throughput_ < 0.01) {
        baseline_throughput_ = gs.total_throughput;
    } else {
        baseline_throughput_ = 0.95 * baseline_throughput_ + 0.05 * gs.total_throughput;
    }

    // Commit snapshots
    {
        std::lock_guard lock(snap_mtx_);
        cached_streams_ = std::move(snaps);
        cached_categories_ = cats;
        cached_global_ = gs;
    }

    prev_data_ = std::move(new_prev);
    prev_sample_time_ = now;

    push_history(gs);
}

void StreamMonitor::check_alerts() {
    std::lock_guard lock(alert_mtx_);
    alerts_.clear();

    auto now = std::chrono::steady_clock::now();

    // Check per-stream stalls
    auto health = orchestrator_.health_check();
    for (auto& h : health) {
        if (h.stalled) {
            alerts_.push_back({
                AlertLevel::Critical, AlertType::Stall, h.id,
                "Stream " + std::to_string(h.id) + " stalled",
                now
            });
        }
    }

    // Throughput drop
    GlobalSnapshot gs;
    {
        std::lock_guard slock(snap_mtx_);
        gs = cached_global_;
    }
    if (baseline_throughput_ > 1.0 && gs.total_throughput < baseline_throughput_ * (1.0 - thresholds_.throughput_drop_pct / 100.0)) {
        alerts_.push_back({
            AlertLevel::Warning, AlertType::ThroughputDrop, 0,
            "Throughput dropped to " + std::to_string(static_cast<int>(gs.total_throughput)) +
            " ticks/s (baseline: " + std::to_string(static_cast<int>(baseline_throughput_)) + ")",
            now
        });
    }

    // High latency
    auto lstats = latency_hist_.compute();
    if (lstats.n > 10 && lstats.p99 > thresholds_.high_latency_us) {
        alerts_.push_back({
            AlertLevel::Warning, AlertType::HighLatency, 0,
            "p99 latency " + std::to_string(static_cast<int>(lstats.p99)) + "us exceeds threshold",
            now
        });
    }
}

void StreamMonitor::push_history(const GlobalSnapshot& gs) {
    auto lstats = latency_hist_.compute();
    HistoryEntry he;
    he.total_throughput = gs.total_throughput;
    he.active_streams = gs.active_streams;
    he.p99_latency_us = lstats.p99;
    he.timestamp = gs.timestamp;

    std::lock_guard lock(hist_mtx_);
    history_buf_[history_write_] = he;
    history_write_ = (history_write_ + 1) % kHistoryCapacity;
    if (history_count_ < kHistoryCapacity) history_count_++;
}

// ─── Public queries ─────────────────────────────────────────────────────────

std::vector<StreamSnapshot> StreamMonitor::stream_snapshots() const {
    std::lock_guard lock(snap_mtx_);
    return cached_streams_;
}

std::array<CategorySnapshot, static_cast<size_t>(StreamCategory::Count)>
StreamMonitor::category_snapshots() const {
    std::lock_guard lock(snap_mtx_);
    return cached_categories_;
}

GlobalSnapshot StreamMonitor::global_snapshot() const {
    std::lock_guard lock(snap_mtx_);
    return cached_global_;
}

LatencyHistogram::Stats StreamMonitor::latency_stats() const {
    return latency_hist_.compute();
}

void StreamMonitor::record_tick_latency(double latency_us) {
    latency_hist_.record(latency_us);
}

std::vector<Alert> StreamMonitor::active_alerts() const {
    std::lock_guard lock(alert_mtx_);
    return alerts_;
}

void StreamMonitor::set_thresholds(AlertThresholds t) {
    std::lock_guard lock(alert_mtx_);
    thresholds_ = std::move(t);
}

std::vector<HistoryEntry> StreamMonitor::history(size_t last_n_seconds) const {
    std::lock_guard lock(hist_mtx_);
    // 10Hz sampling → last_n_seconds * 10 entries
    size_t want = std::min(last_n_seconds * 10, history_count_);
    std::vector<HistoryEntry> result;
    result.reserve(want);

    size_t start;
    if (history_count_ < kHistoryCapacity) {
        start = (history_write_ >= want) ? history_write_ - want : 0;
        want = std::min(want, history_write_);
    } else {
        start = (history_write_ + kHistoryCapacity - want) % kHistoryCapacity;
    }

    for (size_t i = 0; i < want; ++i) {
        result.push_back(history_buf_[(start + i) % kHistoryCapacity]);
    }
    return result;
}

} // namespace brain19
