#pragma once

#include "../common/types.hpp"
#include "../cursor/traversal_types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brain19 {

// Forward declarations
struct ThinkingResult;

// =============================================================================
// GDO CONFIG
// =============================================================================

struct GDOConfig {
    std::chrono::milliseconds tick_interval{500};
    double decay_rate = 0.05;           // Per-tick decay fraction
    double spread_factor = 0.3;         // How much activation spreads to neighbors
    double max_global_energy = 100.0;   // Cap on total accumulated energy
    double thinking_trigger_energy = 30.0; // Energy threshold to trigger autonomous thinking
    double injection_boost = 5.0;       // Default energy per inject_energy() call
    size_t max_activated_concepts = 200; // Max concepts tracked in activation map
    bool enable_autonomous_thinking = true;
};

// =============================================================================
// GDO SNAPSHOT
// =============================================================================

struct GDOSnapshot {
    double global_energy = 0.0;
    size_t active_concepts = 0;
    uint64_t ticks_total = 0;
    uint64_t thinking_cycles_fired = 0;
    std::vector<std::pair<ConceptId, double>> top_activations;
};

// =============================================================================
// GLOBAL DYNAMICS OPERATOR
// =============================================================================
//
// Background process that maintains a global activation landscape.
// Periodically decays activations and spreads energy through the graph.
// Can trigger autonomous thinking when energy accumulates.
//
// Thread-safe: all public methods can be called from any thread.
//

class GlobalDynamicsOperator {
public:
    // Callback for autonomous thinking (called from GDO thread)
    using ThinkingCallback = std::function<void(const std::vector<ConceptId>& seeds)>;

    explicit GlobalDynamicsOperator(GDOConfig config = GDOConfig());
    ~GlobalDynamicsOperator();

    // Non-copyable
    GlobalDynamicsOperator(const GlobalDynamicsOperator&) = delete;
    GlobalDynamicsOperator& operator=(const GlobalDynamicsOperator&) = delete;

    // ── Lifecycle ────────────────────────────────────────────────────────────

    void start();
    void stop();
    bool is_running() const { return running_.load(std::memory_order_acquire); }

    // ── Energy Injection ─────────────────────────────────────────────────────

    // Inject energy (e.g., from user query)
    void inject_energy(double amount);

    // Inject specific concepts as activated
    void inject_seeds(const std::vector<ConceptId>& seeds, double activation = 1.0);

    // ── Feedback ─────────────────────────────────────────────────────────────

    // Feed traversal results back into the activation landscape
    void feed_traversal_result(const TraversalResult& result);

    // ── Observation ──────────────────────────────────────────────────────────

    // Get snapshot of current activation state
    GDOSnapshot get_snapshot(size_t top_k = 10) const;

    // Get top-k most activated concepts
    std::vector<std::pair<ConceptId, double>> get_activation_snapshot(size_t k) const;

    // Get total global energy
    double get_global_energy() const;

    // ── Configuration ────────────────────────────────────────────────────────

    void set_thinking_callback(ThinkingCallback cb);
    const GDOConfig& config() const { return config_; }

private:
    GDOConfig config_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;

    // Activation state (protected by mtx_)
    std::unordered_map<ConceptId, double> activations_;
    double global_energy_ = 0.0;
    uint64_t ticks_total_ = 0;
    uint64_t thinking_cycles_fired_ = 0;

    ThinkingCallback thinking_callback_;

    // ── Internal ─────────────────────────────────────────────────────────────

    void run_loop();
    void tick();
    void decay_activations();
    void prune_activations();
    void maybe_trigger_thinking();

    // Must be called with mtx_ held
    std::vector<std::pair<ConceptId, double>> get_activation_snapshot_impl(size_t k) const;
};

} // namespace brain19
