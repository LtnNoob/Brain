#pragma once

#include <cstdint>
#include <chrono>
#include <thread>

namespace brain19 {

// Subsystem flags — bitfield for which thinking subsystems are active
enum class Subsystem : uint32_t {
    None         = 0,
    Spreading    = 1 << 0,
    Salience     = 1 << 1,
    Curiosity    = 1 << 2,
    Understanding = 1 << 3,
    All          = Spreading | Salience | Curiosity | Understanding
};

constexpr Subsystem operator|(Subsystem a, Subsystem b) {
    return static_cast<Subsystem>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr Subsystem operator&(Subsystem a, Subsystem b) {
    return static_cast<Subsystem>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr bool has_subsystem(Subsystem flags, Subsystem sub) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(sub)) != 0;
}

enum class BackoffStrategy {
    SpinYieldSleep,  // 3-tier: spin N times → yield → sleep
    YieldSleep,      // 2-tier: yield → sleep
    Sleep             // Direct sleep
};

struct StreamConfig {
    // 0 = auto-detect via hardware_concurrency()
    uint32_t max_streams = 0;

    BackoffStrategy backoff_strategy = BackoffStrategy::SpinYieldSleep;

    // Spin count before yielding (tier 1)
    uint32_t spin_count = 100;
    // Yield count before sleeping (tier 2)
    uint32_t yield_count = 10;
    // Sleep duration (tier 3)
    std::chrono::microseconds sleep_duration{500};

    // Tick interval for the thinking cycle
    std::chrono::milliseconds tick_interval{10};

    // Which subsystems each stream runs by default
    Subsystem subsystem_flags = Subsystem::All;

    // Monitor health-check interval
    std::chrono::milliseconds monitor_interval{1000};

    // Graceful shutdown timeout
    std::chrono::seconds shutdown_timeout{5};

    // Stall detection threshold (no progress for this long)
    std::chrono::milliseconds stall_threshold{5000};

    uint32_t effective_max_streams() const {
        if (max_streams > 0) return max_streams;
        auto hw = std::thread::hardware_concurrency();
        return hw > 0 ? hw : 4;
    }
};

} // namespace brain19
