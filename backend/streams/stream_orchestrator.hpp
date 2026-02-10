#pragma once

#include "think_stream.hpp"
#include "stream_config.hpp"
#include "../concurrent/shared_ltm.hpp"
#include "../concurrent/shared_stm.hpp"
#include "../concurrent/shared_registry.hpp"
#include "../concurrent/shared_embeddings.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace brain19 {

// Health status for a stream
struct StreamHealth {
    ThinkStream::StreamId id;
    StreamState state;
    uint64_t total_ticks;
    uint64_t errors;
    bool stalled;
};

// Alert callback
using AlertCallback = std::function<void(const std::string& message)>;

// Orchestrator metrics
struct OrchestratorMetrics {
    std::atomic<uint32_t> active_streams{0};
    std::atomic<uint64_t> total_ticks_all{0};
    std::atomic<uint64_t> total_errors_all{0};
    std::atomic<uint32_t> stalled_streams{0};
};

class StreamOrchestrator {
public:
    StreamOrchestrator(SharedLTM& ltm,
                       SharedSTM& stm,
                       SharedRegistry& registry,
                       SharedEmbeddings& embeddings,
                       const StreamConfig& config = {});

    ~StreamOrchestrator();

    StreamOrchestrator(const StreamOrchestrator&) = delete;
    StreamOrchestrator& operator=(const StreamOrchestrator&) = delete;

    // === Lifecycle ===
    
    // Create a new stream, returns its ID
    ThinkStream::StreamId create_stream();
    ThinkStream::StreamId create_stream(Subsystem subsystems);

    // Start a specific stream
    bool start_stream(ThinkStream::StreamId id);

    // Stop a specific stream
    void stop_stream(ThinkStream::StreamId id);

    // Destroy a stream (stop + remove)
    void destroy_stream(ThinkStream::StreamId id);

    // Start all created streams
    void start_all();

    // Stop all streams gracefully with timeout
    bool shutdown(std::chrono::milliseconds timeout = std::chrono::milliseconds{0});

    // Auto-scale: create and start N streams based on hardware
    void auto_scale();

    // === Monitoring ===

    // Start the monitor thread
    void start_monitor();
    void stop_monitor();

    // Get health of all streams
    std::vector<StreamHealth> health_check() const;

    // Get orchestrator-level metrics
    const OrchestratorMetrics& metrics() const { return metrics_; }

    // Set alert callback
    void set_alert_callback(AlertCallback cb);

    // === Work distribution ===

    // Push a task to the least-loaded stream
    bool distribute_task(ThinkTask task);

    // === Queries ===
    
    uint32_t stream_count() const;
    uint32_t running_count() const;
    const StreamConfig& config() const { return config_; }

private:
    void monitor_loop();
    ThinkStream* find_stream(ThinkStream::StreamId id) const;
    ThinkStream* least_loaded_stream() const;

    SharedLTM& ltm_;
    SharedSTM& stm_;
    SharedRegistry& registry_;
    SharedEmbeddings& embeddings_;
    StreamConfig config_;

    mutable std::mutex streams_mtx_;
    std::unordered_map<ThinkStream::StreamId, std::unique_ptr<ThinkStream>> streams_;
    std::atomic<ThinkStream::StreamId> next_id_{1};

    // Monitor
    std::thread monitor_thread_;
    std::atomic<bool> monitor_running_{false};
    AlertCallback alert_cb_;

    OrchestratorMetrics metrics_;
};

} // namespace brain19
