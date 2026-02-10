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
    if (thread_.joinable()) {
        // Wait with timeout to avoid blocking forever
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            auto s = state_.load(std::memory_order_acquire);
            if (s == StreamState::Stopped || s == StreamState::Error) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto s = state_.load(std::memory_order_acquire);
        if (s == StreamState::Stopped || s == StreamState::Error) {
            thread_.join();
        } else {
            thread_.detach();  // last resort — stop_requested_ is true, thread will exit
        }
    }
    // Destroy our context if we created one
    if (context_id_ != 0) {
        try { stm_.destroy_context(context_id_); } catch (...) {}
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
    if (context_id_ != 0) {
        try { stm_.destroy_context(context_id_); } catch (...) {}
        context_id_ = 0;
    }

    // Create a dedicated STM context for this stream
    context_id_ = stm_.create_context();

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

    // Timeout — thread still running. Detach to avoid blocking destructor forever.
    // The stop flag is already set, so the thread will eventually exit.
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
    auto active = stm_.get_active_concepts(context_id_, 0.3);
    for (auto cid : active) {
        auto relations = ltm_.get_outgoing_relations(cid);
        double source_act = stm_.get_concept_activation(context_id_, cid);
        for (auto& rel : relations) {
            double spread = source_act * rel.weight * 0.5;
            if (spread > 0.05) {
                stm_.boost_concept(context_id_, rel.target, spread);
            }
        }
    }
}

void ThinkStream::do_salience() {
    // Salience: decay activations, prune weak ones
    stm_.decay_all(context_id_, 
        static_cast<double>(config_.tick_interval.count()) / 1000.0);
}

void ThinkStream::do_curiosity() {
    // Curiosity: find concepts with low activation but high connectivity
    // and give them a small boost (exploration)
    auto all_ids = ltm_.get_all_concept_ids();
    for (auto cid : all_ids) {
        double act = stm_.get_concept_activation(context_id_, cid);
        if (act < 0.1) {
            size_t rel_count = ltm_.get_relation_count(cid);
            if (rel_count > 3) {
                double boost = 0.02 * static_cast<double>(std::min(rel_count, size_t(10)));
                stm_.activate_concept(context_id_, cid, boost, ActivationClass::CONTEXTUAL);
            }
        }
    }
}

void ThinkStream::do_understanding() {
    // Understanding: check active concepts for contradictions
    auto active = stm_.get_active_concepts(context_id_, 0.5);
    for (size_t i = 0; i < active.size(); ++i) {
        auto rels = ltm_.get_outgoing_relations(active[i]);
        for (auto& rel : rels) {
            if (rel.type == RelationType::CONTRADICTS) {
                double other_act = stm_.get_concept_activation(context_id_, rel.target);
                if (other_act > 0.5) {
                    // Both contradicting concepts active — dampen both slightly
                    stm_.boost_concept(context_id_, active[i], -0.1);
                    stm_.boost_concept(context_id_, rel.target, -0.1);
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
