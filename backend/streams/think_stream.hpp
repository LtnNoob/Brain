#pragma once

#include "stream_config.hpp"
#include "lock_free_queue.hpp"
#include "../concurrent/shared_ltm.hpp"
#include "../concurrent/shared_stm.hpp"
#include "../concurrent/shared_registry.hpp"
#include "../concurrent/shared_embeddings.hpp"
#include "../common/types.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace brain19 {

// Stream state machine
enum class StreamState : uint32_t {
    Created,
    Starting,
    Running,
    Paused,
    Stopping,
    Stopped,
    Error
};

// Per-stream metrics
struct alignas(64) StreamMetrics {
    std::atomic<uint64_t> total_ticks{0};
    std::atomic<uint64_t> spreading_ticks{0};
    std::atomic<uint64_t> salience_ticks{0};
    std::atomic<uint64_t> curiosity_ticks{0};
    std::atomic<uint64_t> understanding_ticks{0};
    std::atomic<uint64_t> idle_ticks{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<int64_t> last_tick_epoch_us{0};  // for stall detection

    void reset() {
        total_ticks.store(0, std::memory_order_relaxed);
        spreading_ticks.store(0, std::memory_order_relaxed);
        salience_ticks.store(0, std::memory_order_relaxed);
        curiosity_ticks.store(0, std::memory_order_relaxed);
        understanding_ticks.store(0, std::memory_order_relaxed);
        idle_ticks.store(0, std::memory_order_relaxed);
        errors.store(0, std::memory_order_relaxed);
        last_tick_epoch_us.store(0, std::memory_order_relaxed);
    }
};

// A task that can be pushed to a stream's work queue
struct ThinkTask {
    enum class Type : uint8_t { Tick, Custom };
    Type type = Type::Tick;
    ConceptId target_concept = 0;  // optional focus
};

// A single autonomous thinking stream
class ThinkStream {
public:
    using StreamId = uint32_t;

    ThinkStream(StreamId id,
                SharedLTM& ltm,
                SharedSTM& stm,
                SharedRegistry& registry,
                SharedEmbeddings& embeddings,
                const StreamConfig& config);

    ~ThinkStream();

    // Non-copyable, non-movable (owns thread)
    ThinkStream(const ThinkStream&) = delete;
    ThinkStream& operator=(const ThinkStream&) = delete;

    // Lifecycle
    bool start();
    void stop();       // request stop
    bool join(std::chrono::milliseconds timeout);  // wait for thread to finish

    // Task injection (work-stealing target)
    bool push_task(ThinkTask task);

    // State
    StreamId id() const { return id_; }
    ContextId context_id() const { return context_id_; }
    StreamState state() const { return state_.load(std::memory_order_acquire); }
    const StreamMetrics& metrics() const { return metrics_; }
    size_t pending_tasks() const { return work_queue_.size_approx(); }

    // Configuration
    void set_subsystems(Subsystem flags);
    Subsystem subsystems() const { return subsystems_.load(std::memory_order_relaxed); }

private:
    void run();
    void tick();
    void do_spreading();
    void do_salience();
    void do_curiosity();
    void do_understanding();
    void backoff(uint32_t& idle_count);

    const StreamId id_;
    ContextId context_id_ = 0;

    SharedLTM& ltm_;
    SharedSTM& stm_;
    SharedRegistry& registry_;
    SharedEmbeddings& embeddings_;

    StreamConfig config_;
    std::atomic<Subsystem> subsystems_;
    std::atomic<StreamState> state_{StreamState::Created};
    std::atomic<bool> stop_requested_{false};

    MPMCQueue<ThinkTask> work_queue_;
    StreamMetrics metrics_;

    size_t curiosity_offset_ = 0;  // Round-robin offset for do_curiosity sampling

    std::thread thread_;
};

} // namespace brain19
