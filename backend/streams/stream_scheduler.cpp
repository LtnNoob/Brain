#include "stream_scheduler.hpp"
#include <algorithm>
#include <cmath>

namespace brain19 {

StreamScheduler::StreamScheduler(StreamOrchestrator& orchestrator, SchedulerConfig config)
    : orchestrator_(orchestrator)
    , config_(std::move(config))
{}

StreamScheduler::~StreamScheduler() {
    shutdown(std::chrono::milliseconds{3000});
}

// === Lifecycle ===

void StreamScheduler::start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) return;

    // Create initial streams per category budget defaults
    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        auto cat = static_cast<StreamCategory>(i);
        auto budget = config_.budgets[i];
        for (uint32_t s = 0; s < budget.default_streams; ++s) {
            create_categorized_stream(cat);
        }
    }

    // Start rebalance thread
    rebalance_thread_ = std::thread(&StreamScheduler::rebalance_loop, this);
}

bool StreamScheduler::shutdown(std::chrono::milliseconds timeout) {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return true;

    if (rebalance_thread_.joinable()) {
        rebalance_thread_.join();
    }

    // Stop all categorized streams
    {
        std::lock_guard lock(cat_mtx_);
        for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
            for (auto sid : category_streams_[i]) {
                orchestrator_.stop_stream(sid);
            }
            category_streams_[i].clear();
            stats_[i].active_streams.store(0, std::memory_order_relaxed);
        }
    }

    return orchestrator_.shutdown(timeout);
}

// === Category management ===

uint32_t StreamScheduler::create_categorized_stream(StreamCategory cat) {
    auto idx = static_cast<size_t>(cat);
    auto cfg_subsystems = category_subsystems(cat);

    auto sid = orchestrator_.create_stream(cfg_subsystems);
    orchestrator_.start_stream(sid);

    {
        std::lock_guard lock(cat_mtx_);
        category_streams_[idx].push_back(sid);
        stats_[idx].active_streams.fetch_add(1, std::memory_order_relaxed);
    }

    return sid;
}

void StreamScheduler::destroy_categorized_stream(StreamCategory cat, uint32_t stream_id) {
    auto idx = static_cast<size_t>(cat);

    {
        std::lock_guard lock(cat_mtx_);
        auto& vec = category_streams_[idx];
        vec.erase(std::remove(vec.begin(), vec.end(), stream_id), vec.end());
        auto prev = stats_[idx].active_streams.load(std::memory_order_relaxed);
        if (prev > 0) stats_[idx].active_streams.fetch_sub(1, std::memory_order_relaxed);
    }

    orchestrator_.destroy_stream(stream_id);
}

std::vector<uint32_t> StreamScheduler::get_streams(StreamCategory cat) const {
    std::lock_guard lock(cat_mtx_);
    return category_streams_[static_cast<size_t>(cat)];
}

uint32_t StreamScheduler::stream_count(StreamCategory cat) const {
    std::lock_guard lock(cat_mtx_);
    return static_cast<uint32_t>(category_streams_[static_cast<size_t>(cat)].size());
}

// === Scheduling ===

bool StreamScheduler::schedule_task(StreamCategory cat, ThinkTask task) {
    auto idx = static_cast<size_t>(cat);
    stats_[idx].total_scheduled.fetch_add(1, std::memory_order_relaxed);
    stats_[idx].starvation_count.store(0, std::memory_order_relaxed);

    // Try streams in this category
    std::lock_guard lock(cat_mtx_);
    for (auto sid : category_streams_[idx]) {
        // Use orchestrator's distribute mechanism via direct push
        // We just need any stream in this category to accept
        if (orchestrator_.distribute_task(std::move(task))) {
            return true;
        }
    }
    return false;
}

bool StreamScheduler::schedule_task_by_priority(ThinkTask task) {
    // Try categories in priority order: Reasoning(0) > Perception(1) > Creative(2) > Memory(3)
    // Build sorted order
    std::array<StreamCategory, static_cast<size_t>(StreamCategory::Count)> order = {
        StreamCategory::Reasoning,
        StreamCategory::Perception,
        StreamCategory::Creative,
        StreamCategory::Memory
    };

    // Fair scheduling: if any category has been starved too long, bump it up
    for (auto cat : order) {
        auto idx = static_cast<size_t>(cat);
        auto starved = stats_[idx].starvation_count.load(std::memory_order_relaxed);
        if (starved >= config_.max_starvation_rounds) {
            // Force schedule to this starved category
            if (schedule_task(cat, std::move(task))) return true;
        }
    }

    // Normal priority scheduling
    for (auto cat : order) {
        auto idx = static_cast<size_t>(cat);
        std::lock_guard lock(cat_mtx_);
        if (!category_streams_[idx].empty()) {
            // Increment starvation for categories we skip
            for (auto other : order) {
                if (other != cat) {
                    stats_[static_cast<size_t>(other)].starvation_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
            // Reset starvation for scheduled category
            stats_[idx].starvation_count.store(0, std::memory_order_relaxed);
            stats_[idx].total_scheduled.fetch_add(1, std::memory_order_relaxed);

            if (orchestrator_.distribute_task(std::move(task))) {
                return true;
            }
        }
    }

    return false;
}

// === Dynamic reallocation ===

void StreamScheduler::update_load(const SystemLoad& load) {
    std::lock_guard lock(load_mtx_);
    current_load_ = load;
}

void StreamScheduler::rebalance() {
    SystemLoad load;
    {
        std::lock_guard lock(load_mtx_);
        load = current_load_;
    }

    for (size_t i = 0; i < static_cast<size_t>(StreamCategory::Count); ++i) {
        auto cat = static_cast<StreamCategory>(i);
        uint32_t target = compute_target_streams(cat, load);
        scale_category(cat, target);
    }
}

uint32_t StreamScheduler::compute_target_streams(StreamCategory cat, const SystemLoad& load) const {
    auto idx = static_cast<size_t>(cat);
    auto budget = config_.budgets[idx];

    uint32_t target = budget.default_streams;

    switch (cat) {
        case StreamCategory::Reasoning:
            // More reasoning streams when high activation
            if (load.activation_level > 0.7) {
                target = budget.max_streams;
            } else if (load.activation_level > 0.4) {
                target = (budget.min_streams + budget.max_streams) / 2;
            }
            break;

        case StreamCategory::Perception:
            // More perception streams when inputs pending
            if (load.pending_inputs > 10) {
                target = budget.max_streams;
            } else if (load.pending_inputs > 3) {
                target = (budget.min_streams + budget.max_streams) / 2;
            }
            break;

        case StreamCategory::Memory:
            // More memory streams when idle
            if (load.idle_ratio > 0.7) {
                target = budget.max_streams;
            }
            break;

        case StreamCategory::Creative:
            // More creative streams when moderately idle
            if (load.idle_ratio > 0.5 && load.activation_level > 0.2) {
                target = budget.max_streams;
            }
            break;

        default:
            break;
    }

    // Enforce budget bounds
    target = std::clamp(target, budget.min_streams, budget.max_streams);

    // Enforce total budget
    uint32_t total_others = 0;
    {
        std::lock_guard lock(cat_mtx_);
        for (size_t j = 0; j < static_cast<size_t>(StreamCategory::Count); ++j) {
            if (j != idx) {
                total_others += static_cast<uint32_t>(category_streams_[j].size());
            }
        }
    }
    uint32_t total_max = config_.effective_total_max();
    if (total_others + target > total_max) {
        target = (total_max > total_others) ? (total_max - total_others) : budget.min_streams;
    }

    return target;
}

void StreamScheduler::scale_category(StreamCategory cat, uint32_t target) {
    uint32_t current = stream_count(cat);

    if (target > current) {
        for (uint32_t i = 0; i < (target - current); ++i) {
            create_categorized_stream(cat);
        }
    } else if (target < current) {
        // Remove excess streams (LIFO)
        uint32_t to_remove = current - target;
        for (uint32_t i = 0; i < to_remove; ++i) {
            uint32_t sid = 0;
            {
                std::lock_guard lock(cat_mtx_);
                auto& vec = category_streams_[static_cast<size_t>(cat)];
                if (vec.empty()) break;
                sid = vec.back();
            }
            destroy_categorized_stream(cat, sid);
        }
    }
}

void StreamScheduler::rebalance_loop() {
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(config_.rebalance_interval);
        if (!running_.load(std::memory_order_acquire)) break;
        rebalance();
    }
}

} // namespace brain19
