#include "think_stream.hpp"
#include <algorithm>
#include <chrono>

namespace brain19 {

ThinkStream::ThinkStream(StreamId id,
                         SharedLTM& ltm,
                         SharedSTM& stm,
                         SharedRegistry& registry,
                         SharedEmbeddings& embeddings,
                         const StreamConfig& config)
    : id_(id)
    , ltm_(ltm)
    , stm_(stm)
    , registry_(registry)
    , embeddings_(embeddings)
    , config_(config)
    , subsystems_(config.subsystem_flags)
    , work_queue_(1024)
{}

ThinkStream::~ThinkStream() {
    stop();
    // Never detach — the thread reads our members, so we must outlive it.
    // stop_requested_ is set, so the thread will exit its run-loop.
    if (thread_.joinable()) {
        thread_.join();
    }
    // Destroy our context if we created one
    auto cid = context_id_.load(std::memory_order_acquire);
    if (cid != 0) {
        try { stm_.destroy_context(cid); } catch (...) {}
    }
}

bool ThinkStream::start() {
    StreamState expected = StreamState::Created;
    if (!state_.compare_exchange_strong(expected, StreamState::Starting,
                                         std::memory_order_acq_rel)) {
        // Also allow restart from Stopped
        expected = StreamState::Stopped;
        if (!state_.compare_exchange_strong(expected, StreamState::Starting,
                                             std::memory_order_acq_rel)) {
            return false;
        }
    }

    // Join old thread if still joinable (restart from Stopped)
    if (thread_.joinable()) {
        thread_.join();
    }

    stop_requested_.store(false, std::memory_order_relaxed);
    metrics_.reset();

    // Destroy old context if exists (prevent context leak on restart)
    auto old_cid = context_id_.load(std::memory_order_relaxed);
    if (old_cid != 0) {
        try { stm_.destroy_context(old_cid); } catch (...) {}
        context_id_.store(0, std::memory_order_relaxed);
    }

    // Create a dedicated STM context for this stream
    context_id_.store(stm_.create_context(), std::memory_order_release);

    thread_ = std::thread(&ThinkStream::run, this);
    return true;
}

void ThinkStream::stop() {
    stop_requested_.store(true, std::memory_order_release);
}

bool ThinkStream::join(std::chrono::milliseconds timeout) {
    if (!thread_.joinable()) return true;

    // Busy-wait with timeout
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        auto s = state_.load(std::memory_order_acquire);
        if (s == StreamState::Stopped || s == StreamState::Error) {
            thread_.join();
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Timeout — thread still running but stop_requested_ is set.
    // Do NOT detach (would cause use-after-free). Caller should retry or
    // let the destructor block until the thread exits.
    return false;
}

bool ThinkStream::push_task(ThinkTask task) {
    return work_queue_.try_push(std::move(task));
}

void ThinkStream::set_subsystems(Subsystem flags) {
    subsystems_.store(flags, std::memory_order_relaxed);
}

void ThinkStream::run() {
    state_.store(StreamState::Running, std::memory_order_release);
    uint32_t idle_count = 0;

    while (!stop_requested_.load(std::memory_order_acquire)) {
        try {
            // Check for injected tasks first
            auto task = work_queue_.try_pop();
            if (task.has_value()) {
                idle_count = 0;
                if (task->type == ThinkTask::Type::Tick) {
                    tick();
                }
                // Custom tasks could be extended here
            } else {
                // No external task — run autonomous tick
                tick();
                idle_count = 0;

                // Sleep for tick interval
                std::this_thread::sleep_for(config_.tick_interval);
            }
        } catch (...) {
            metrics_.errors.fetch_add(1, std::memory_order_relaxed);
            // Don't crash the stream — backoff and retry
            backoff(idle_count);
        }
    }

    state_.store(StreamState::Stopped, std::memory_order_release);
}

void ThinkStream::tick() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    metrics_.last_tick_epoch_us.store(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count(),
        std::memory_order_relaxed);
    metrics_.total_ticks.fetch_add(1, std::memory_order_relaxed);

    Subsystem flags = subsystems_.load(std::memory_order_relaxed);

    if (has_subsystem(flags, Subsystem::Spreading)) {
        do_spreading();
        metrics_.spreading_ticks.fetch_add(1, std::memory_order_relaxed);
    }
    if (has_subsystem(flags, Subsystem::Salience)) {
        do_salience();
        metrics_.salience_ticks.fetch_add(1, std::memory_order_relaxed);
    }
    if (has_subsystem(flags, Subsystem::Curiosity)) {
        do_curiosity();
        metrics_.curiosity_ticks.fetch_add(1, std::memory_order_relaxed);
    }
    if (has_subsystem(flags, Subsystem::Understanding)) {
        do_understanding();
        metrics_.understanding_ticks.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Subsystem implementations — these are the actual "thinking" operations.
// They read/write shared state through the thread-safe wrappers.
// ---------------------------------------------------------------------------

void ThinkStream::do_spreading() {
    // Spreading activation: take active concepts from STM,
    // follow LTM relations, boost neighbors
    auto ctx = context_id_.load(std::memory_order_acquire);
    if (ctx == 0) return;
    auto active = stm_.get_active_concepts(ctx, 0.3);
    for (auto concept_id : active) {
        auto relations = ltm_.get_outgoing_relations(concept_id);
        double source_act = stm_.get_concept_activation(ctx, concept_id);
        for (auto& rel : relations) {
            double spread = source_act * rel.weight * 0.5;
            if (spread > 0.05) {
                stm_.boost_concept(ctx, rel.target, spread);
            }
        }
    }
}

void ThinkStream::do_salience() {
    // Salience: decay activations, prune weak ones
    auto ctx = context_id_.load(std::memory_order_acquire);
    if (ctx == 0) return;
    stm_.decay_all(ctx,
        static_cast<double>(config_.tick_interval.count()) / 1000.0);
}

void ThinkStream::do_curiosity() {
    // Curiosity: find concepts with low activation but high connectivity
    // and give them a small boost (exploration)
    // Round-robin sampling to avoid iterating all concepts every tick
    constexpr size_t max_per_tick = 64;
    auto all_ids = ltm_.get_all_concept_ids();
    if (all_ids.empty()) return;
    if (curiosity_offset_ >= all_ids.size()) curiosity_offset_ = 0;
    size_t end = std::min(curiosity_offset_ + max_per_tick, all_ids.size());
    auto ctx = context_id_.load(std::memory_order_acquire);
    if (ctx == 0) return;
    for (size_t i = curiosity_offset_; i < end; ++i) {
        auto concept_id = all_ids[i];
        double act = stm_.get_concept_activation(ctx, concept_id);
        if (act < 0.1) {
            size_t rel_count = ltm_.get_relation_count(concept_id);
            if (rel_count > 3) {
                double boost = 0.02 * static_cast<double>(std::min(rel_count, size_t(10)));
                stm_.activate_concept(ctx, concept_id, boost, ActivationClass::CONTEXTUAL);
            }
        }
    }
    curiosity_offset_ = end;
}

void ThinkStream::do_understanding() {
    // Understanding: check active concepts for contradictions
    auto ctx = context_id_.load(std::memory_order_acquire);
    if (ctx == 0) return;
    auto active = stm_.get_active_concepts(ctx, 0.5);
    for (size_t i = 0; i < active.size(); ++i) {
        auto rels = ltm_.get_outgoing_relations(active[i]);
        for (auto& rel : rels) {
            if (rel.type == RelationType::CONTRADICTS) {
                double other_act = stm_.get_concept_activation(ctx, rel.target);
                if (other_act > 0.5) {
                    // Both contradicting concepts active — dampen both slightly
                    stm_.boost_concept(ctx, active[i], -0.1);
                    stm_.boost_concept(ctx, rel.target, -0.1);
                }
            }
        }
    }
}

void ThinkStream::backoff(uint32_t& idle_count) {
    switch (config_.backoff_strategy) {
        case BackoffStrategy::SpinYieldSleep:
            if (idle_count < config_.spin_count) {
                ++idle_count;
                // spin — do nothing
            } else if (idle_count < config_.spin_count + config_.yield_count) {
                ++idle_count;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(config_.sleep_duration);
            }
            break;
        case BackoffStrategy::YieldSleep:
            if (idle_count < config_.yield_count) {
                ++idle_count;
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(config_.sleep_duration);
            }
            break;
        case BackoffStrategy::Sleep:
            std::this_thread::sleep_for(config_.sleep_duration);
            break;
    }
}

} // namespace brain19
