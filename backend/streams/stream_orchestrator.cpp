#include "stream_orchestrator.hpp"
#include <algorithm>
#include <iostream>

namespace brain19 {

StreamOrchestrator::StreamOrchestrator(SharedLTM& ltm,
                                       SharedSTM& stm,
                                       SharedRegistry& registry,
                                       SharedEmbeddings& embeddings,
                                       const StreamConfig& config)
    : ltm_(ltm), stm_(stm), registry_(registry), embeddings_(embeddings), config_(config)
{}

StreamOrchestrator::~StreamOrchestrator() {
    stop_monitor();
    shutdown(std::chrono::milliseconds(config_.shutdown_timeout));
}

// === Lifecycle ===

ThinkStream::StreamId StreamOrchestrator::create_stream() {
    return create_stream(config_.subsystem_flags);
}

ThinkStream::StreamId StreamOrchestrator::create_stream(Subsystem subsystems) {
    auto id = next_id_.fetch_add(1, std::memory_order_relaxed);
    auto stream = std::make_unique<ThinkStream>(id, ltm_, stm_, registry_, embeddings_, config_);
    stream->set_subsystems(subsystems);

    std::lock_guard lock(streams_mtx_);
    streams_[id] = std::move(stream);
    return id;
}

bool StreamOrchestrator::start_stream(ThinkStream::StreamId id) {
    std::lock_guard lock(streams_mtx_);
    auto it = streams_.find(id);
    if (it == streams_.end()) return false;
    bool ok = it->second->start();
    if (ok) {
        metrics_.active_streams.fetch_add(1, std::memory_order_relaxed);
    }
    return ok;
}

void StreamOrchestrator::stop_stream(ThinkStream::StreamId id) {
    std::lock_guard lock(streams_mtx_);
    auto it = streams_.find(id);
    if (it == streams_.end()) return;
    it->second->stop();
    if (it->second->join(std::chrono::milliseconds(config_.shutdown_timeout))) {
        auto prev = metrics_.active_streams.load(std::memory_order_relaxed);
        if (prev > 0) metrics_.active_streams.fetch_sub(1, std::memory_order_relaxed);
    }
}

void StreamOrchestrator::destroy_stream(ThinkStream::StreamId id) {
    std::unique_ptr<ThinkStream> stream;
    {
        std::lock_guard lock(streams_mtx_);
        auto it = streams_.find(id);
        if (it == streams_.end()) return;
        stream = std::move(it->second);
        streams_.erase(it);
    }
    // Stop and join outside lock
    stream->stop();
    stream->join(std::chrono::milliseconds(config_.shutdown_timeout));
    auto prev = metrics_.active_streams.load(std::memory_order_relaxed);
    if (prev > 0) metrics_.active_streams.fetch_sub(1, std::memory_order_relaxed);
    // stream destroyed here by unique_ptr
}

void StreamOrchestrator::start_all() {
    std::lock_guard lock(streams_mtx_);
    for (auto& [id, stream] : streams_) {
        if (stream->state() == StreamState::Created || 
            stream->state() == StreamState::Stopped) {
            if (stream->start()) {
                metrics_.active_streams.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

bool StreamOrchestrator::shutdown(std::chrono::milliseconds timeout) {
    if (timeout.count() == 0) {
        timeout = std::chrono::milliseconds(config_.shutdown_timeout);
    }

    // Request all to stop
    {
        std::lock_guard lock(streams_mtx_);
        for (auto& [id, stream] : streams_) {
            stream->stop();
        }
    }

    // Join all with timeout
    bool all_stopped = true;
    {
        std::lock_guard lock(streams_mtx_);
        for (auto& [id, stream] : streams_) {
            if (!stream->join(timeout)) {
                all_stopped = false;
            }
        }
    }

    metrics_.active_streams.store(0, std::memory_order_relaxed);
    return all_stopped;
}

void StreamOrchestrator::auto_scale() {
    uint32_t n = config_.effective_max_streams();
    // Don't create more than we already have
    uint32_t current = stream_count();
    for (uint32_t i = current; i < n; ++i) {
        auto id = create_stream();
        start_stream(id);
    }
}

// === Monitoring ===

void StreamOrchestrator::start_monitor() {
    if (monitor_running_.exchange(true, std::memory_order_acq_rel)) return;
    monitor_thread_ = std::thread(&StreamOrchestrator::monitor_loop, this);
}

void StreamOrchestrator::stop_monitor() {
    monitor_running_.store(false, std::memory_order_release);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

std::vector<StreamHealth> StreamOrchestrator::health_check() const {
    std::vector<StreamHealth> result;
    std::lock_guard lock(streams_mtx_);

    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    uint64_t total_ticks = 0;
    uint64_t total_errors = 0;
    uint32_t stalled = 0;

    for (auto& [id, stream] : streams_) {
        auto& m = stream->metrics();
        auto last_us = m.last_tick_epoch_us.load(std::memory_order_relaxed);
        bool is_stalled = (stream->state() == StreamState::Running) &&
                          (last_us > 0) &&
                          ((now_us - last_us) > config_.stall_threshold.count() * 1000);

        StreamHealth h;
        h.id = id;
        h.state = stream->state();
        h.total_ticks = m.total_ticks.load(std::memory_order_relaxed);
        h.errors = m.errors.load(std::memory_order_relaxed);
        h.stalled = is_stalled;

        total_ticks += h.total_ticks;
        total_errors += h.errors;
        if (is_stalled) ++stalled;

        result.push_back(h);
    }

    // Update orchestrator metrics (const_cast for atomic updates from const method)
    const_cast<OrchestratorMetrics&>(metrics_).total_ticks_all.store(total_ticks, std::memory_order_relaxed);
    const_cast<OrchestratorMetrics&>(metrics_).total_errors_all.store(total_errors, std::memory_order_relaxed);
    const_cast<OrchestratorMetrics&>(metrics_).stalled_streams.store(stalled, std::memory_order_relaxed);

    return result;
}

void StreamOrchestrator::set_alert_callback(AlertCallback cb) {
    alert_cb_ = std::move(cb);
}

bool StreamOrchestrator::distribute_task(ThinkTask task) {
    std::lock_guard lock(streams_mtx_);
    auto* stream = least_loaded_stream();
    if (!stream) return false;
    return stream->push_task(std::move(task));
}

uint32_t StreamOrchestrator::stream_count() const {
    std::lock_guard lock(streams_mtx_);
    return static_cast<uint32_t>(streams_.size());
}

uint32_t StreamOrchestrator::running_count() const {
    std::lock_guard lock(streams_mtx_);
    uint32_t count = 0;
    for (auto& [id, stream] : streams_) {
        if (stream->state() == StreamState::Running) ++count;
    }
    return count;
}

void StreamOrchestrator::monitor_loop() {
    while (monitor_running_.load(std::memory_order_acquire)) {
        auto health = health_check();

        for (auto& h : health) {
            if (h.stalled && alert_cb_) {
                alert_cb_("Stream " + std::to_string(h.id) + " is stalled!");
            }
            if (h.state == StreamState::Error && alert_cb_) {
                alert_cb_("Stream " + std::to_string(h.id) + " in error state!");
            }
        }

        std::this_thread::sleep_for(config_.monitor_interval);
    }
}

ThinkStream* StreamOrchestrator::find_stream(ThinkStream::StreamId id) const {
    auto it = streams_.find(id);
    return it != streams_.end() ? it->second.get() : nullptr;
}

ThinkStream* StreamOrchestrator::least_loaded_stream() const {
    ThinkStream* best = nullptr;
    uint64_t min_ticks = UINT64_MAX;
    for (auto& [id, stream] : streams_) {
        if (stream->state() != StreamState::Running) continue;
        uint64_t t = stream->metrics().total_ticks.load(std::memory_order_relaxed);
        if (t < min_ticks) {
            min_ticks = t;
            best = stream.get();
        }
    }
    return best;
}

} // namespace brain19
