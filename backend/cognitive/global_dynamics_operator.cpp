#include "global_dynamics_operator.hpp"

#include <algorithm>
#include <cmath>

namespace brain19 {

GlobalDynamicsOperator::GlobalDynamicsOperator(GDOConfig config)
    : config_(std::move(config))
{}

GlobalDynamicsOperator::~GlobalDynamicsOperator() {
    stop();
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

void GlobalDynamicsOperator::start() {
    if (running_.load(std::memory_order_acquire)) return;
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this]() { run_loop(); });
}

void GlobalDynamicsOperator::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

// ── Energy Injection ─────────────────────────────────────────────────────────

void GlobalDynamicsOperator::inject_energy(double amount) {
    std::lock_guard<std::mutex> lock(mtx_);
    global_energy_ = std::min(global_energy_ + amount, config_.max_global_energy);
}

void GlobalDynamicsOperator::inject_seeds(const std::vector<ConceptId>& seeds, double activation) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (ConceptId cid : seeds) {
        auto& val = activations_[cid];
        val = std::min(val + activation, 1.0);
    }
    global_energy_ = std::min(global_energy_ + activation * seeds.size(), config_.max_global_energy);
}

// ── Feedback ─────────────────────────────────────────────────────────────────

void GlobalDynamicsOperator::feed_traversal_result(const TraversalResult& result) {
    std::lock_guard<std::mutex> lock(mtx_);
    double weight = result.chain_score;
    for (const auto& step : result.chain) {
        auto& val = activations_[step.concept_id];
        val = std::min(val + weight * step.weight_at_entry, 1.0);
        weight *= 0.8;  // Decay along chain
    }
}

// ── Observation ──────────────────────────────────────────────────────────────

GDOSnapshot GlobalDynamicsOperator::get_snapshot(size_t top_k) const {
    std::lock_guard<std::mutex> lock(mtx_);
    GDOSnapshot snap;
    snap.global_energy = global_energy_;
    snap.active_concepts = activations_.size();
    snap.ticks_total = ticks_total_;
    snap.thinking_cycles_fired = thinking_cycles_fired_;
    snap.top_activations = get_activation_snapshot_impl(top_k);
    return snap;
}

std::vector<std::pair<ConceptId, double>> GlobalDynamicsOperator::get_activation_snapshot(size_t k) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return get_activation_snapshot_impl(k);
}

double GlobalDynamicsOperator::get_global_energy() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return global_energy_;
}

void GlobalDynamicsOperator::set_thinking_callback(ThinkingCallback cb) {
    std::lock_guard<std::mutex> lock(mtx_);
    thinking_callback_ = std::move(cb);
}

// ── Internal Loop ────────────────────────────────────────────────────────────

void GlobalDynamicsOperator::run_loop() {
    while (running_.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait_for(lock, config_.tick_interval, [this]() {
                return !running_.load(std::memory_order_relaxed);
            });
        }
        if (!running_.load(std::memory_order_acquire)) break;
        tick();
    }
}

void GlobalDynamicsOperator::tick() {
    std::lock_guard<std::mutex> lock(mtx_);
    ++ticks_total_;

    decay_activations();
    prune_activations();

    // Decay global energy too
    global_energy_ *= (1.0 - config_.decay_rate * 0.5);
    if (global_energy_ < 0.01) global_energy_ = 0.0;

    maybe_trigger_thinking();
}

void GlobalDynamicsOperator::decay_activations() {
    double decay = config_.decay_rate;
    auto it = activations_.begin();
    while (it != activations_.end()) {
        it->second *= (1.0 - decay);
        if (it->second < 0.001) {
            it = activations_.erase(it);
        } else {
            ++it;
        }
    }
}

void GlobalDynamicsOperator::prune_activations() {
    if (activations_.size() <= config_.max_activated_concepts) return;

    // Keep only top-N by activation
    std::vector<std::pair<ConceptId, double>> sorted(activations_.begin(), activations_.end());
    std::partial_sort(sorted.begin(),
                      sorted.begin() + config_.max_activated_concepts,
                      sorted.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    activations_.clear();
    for (size_t i = 0; i < config_.max_activated_concepts && i < sorted.size(); ++i) {
        activations_[sorted[i].first] = sorted[i].second;
    }
}

void GlobalDynamicsOperator::maybe_trigger_thinking() {
    if (!config_.enable_autonomous_thinking) return;
    if (global_energy_ < config_.thinking_trigger_energy) return;
    if (!thinking_callback_) return;

    // Collect top seeds for thinking
    auto seeds_with_score = get_activation_snapshot_impl(5);
    if (seeds_with_score.empty()) return;

    std::vector<ConceptId> seeds;
    seeds.reserve(seeds_with_score.size());
    for (const auto& [cid, _] : seeds_with_score) {
        seeds.push_back(cid);
    }

    // Consume energy
    global_energy_ *= 0.3;  // Keep some residual
    ++thinking_cycles_fired_;

    // Release lock before callback (callback may re-enter inject_seeds etc.)
    ThinkingCallback cb = thinking_callback_;
    mtx_.unlock();
    try {
        cb(seeds);
    } catch (...) {
        mtx_.lock();
        throw;
    }
    mtx_.lock();
}

// Helper (must be called with mtx_ held)
std::vector<std::pair<ConceptId, double>>
GlobalDynamicsOperator::get_activation_snapshot_impl(size_t k) const {
    std::vector<std::pair<ConceptId, double>> sorted(activations_.begin(), activations_.end());
    if (k < sorted.size()) {
        std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        sorted.resize(k);
    } else {
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }
    return sorted;
}

} // namespace brain19
