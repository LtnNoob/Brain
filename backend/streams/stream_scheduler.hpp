#pragma once

#include "stream_categories.hpp"
#include "stream_orchestrator.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace brain19 {

// Load signal for dynamic reallocation
struct SystemLoad {
    double activation_level = 0.0;   // 0..1 how much activation in STM
    double idle_ratio = 0.0;         // 0..1 how idle the system is
    size_t pending_inputs = 0;       // queued input count
};

// Per-category runtime stats
struct CategoryStats {
    std::atomic<uint32_t> active_streams{0};
    std::atomic<uint64_t> total_ticks{0};
    std::atomic<uint64_t> total_scheduled{0};     // times this category was scheduled
    std::atomic<uint64_t> starvation_count{0};     // times skipped consecutively

    void reset() {
        active_streams.store(0, std::memory_order_relaxed);
        total_ticks.store(0, std::memory_order_relaxed);
        total_scheduled.store(0, std::memory_order_relaxed);
        starvation_count.store(0, std::memory_order_relaxed);
    }
};

// Scheduler configuration
struct SchedulerConfig {
    // Budget overrides (if zero, use category_budget defaults)
    std::array<CategoryBudget, static_cast<size_t>(StreamCategory::Count)> budgets{};

    // Fair scheduling: max consecutive skips before forced schedule
    uint32_t max_starvation_rounds = 5;

    // Rebalance interval
    std::chrono::milliseconds rebalance_interval{500};

    // Total stream budget (0 = auto from hardware)
    uint32_t total_max_streams = 0;

    uint32_t effective_total_max() const {
        if (total_max_streams > 0) return total_max_streams;
        auto hw = std::thread::hardware_concurrency();
        return hw > 0 ? hw : 8;
    }

    SchedulerConfig() {
        for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
            budgets[i] = category_budget(static_cast<StreamCategory>(i));
        }
    }
};

class StreamScheduler {
public:
    StreamScheduler(StreamOrchestrator& orchestrator, SchedulerConfig config = {});
    ~StreamScheduler();

    StreamScheduler(const StreamScheduler&) = delete;
    StreamScheduler& operator=(const StreamScheduler&) = delete;

    // === Lifecycle ===

    // Start the scheduler: creates initial streams per category budgets, starts rebalancer
    void start();

    // Graceful shutdown: stops rebalancer, shuts down all categorized streams
    bool shutdown(std::chrono::milliseconds timeout = std::chrono::milliseconds{5000});

    // === Category management ===

    // Create a stream in a specific category
    uint32_t create_categorized_stream(StreamCategory cat);

    // Destroy a categorized stream
    void destroy_categorized_stream(StreamCategory cat, uint32_t stream_id);

    // Get stream IDs for a category
    std::vector<uint32_t> get_streams(StreamCategory cat) const;

    // Get count for a category
    uint32_t stream_count(StreamCategory cat) const;

    // === Scheduling ===

    // Distribute a task to the best stream in a category
    bool schedule_task(StreamCategory cat, ThinkTask task);

    // Distribute a task using priority scheduling (tries highest priority first)
    bool schedule_task_by_priority(ThinkTask task);

    // === Dynamic reallocation ===

    // Update system load signal (called externally)
    void update_load(const SystemLoad& load);

    // Force rebalance now
    void rebalance();

    // === Stats ===

    const CategoryStats& stats(StreamCategory cat) const {
        return stats_[static_cast<size_t>(cat)];
    }

    const SchedulerConfig& config() const { return config_; }

    bool is_running() const { return running_.load(std::memory_order_acquire); }

private:
    void rebalance_loop();
    void apply_budgets();
    void ensure_min_streams(StreamCategory cat);
    void scale_category(StreamCategory cat, uint32_t target);
    uint32_t compute_target_streams(StreamCategory cat, const SystemLoad& load) const;

    StreamOrchestrator& orchestrator_;
    SchedulerConfig config_;

    mutable std::mutex cat_mtx_;
    std::array<std::vector<uint32_t>, static_cast<size_t>(StreamCategory::Count)> category_streams_;
    std::array<CategoryStats, static_cast<size_t>(StreamCategory::Count)> stats_{};

    SystemLoad current_load_;
    std::mutex load_mtx_;

    std::thread rebalance_thread_;
    std::atomic<bool> running_{false};
};

} // namespace brain19
