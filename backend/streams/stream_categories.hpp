#pragma once

#include "stream_config.hpp"
#include <cstdint>
#include <string>
#include <string_view>

namespace brain19 {

// =============================================================================
// STREAM CATEGORIES (Phase 5.2)
// =============================================================================
// Specialized stream types that configure ThinkStream with different subsystem
// profiles and priorities. Rather than subclassing ThinkStream (which owns a
// thread and is non-movable), categories are lightweight descriptors that
// produce the right StreamConfig + Subsystem flags for the orchestrator.

enum class StreamCategory : uint8_t {
    Perception = 0,  // Input processing: ingestion, entity/relation extraction
    Reasoning  = 1,  // Core thinking: spreading activation, salience, thought paths
    Memory     = 2,  // Consolidation: STM→LTM, decay, micro-model training
    Creative   = 3,  // Exploration: curiosity triggers, analogies, hypotheses
    Count      = 4
};

constexpr std::string_view category_name(StreamCategory cat) {
    switch (cat) {
        case StreamCategory::Perception: return "Perception";
        case StreamCategory::Reasoning:  return "Reasoning";
        case StreamCategory::Memory:     return "Memory";
        case StreamCategory::Creative:   return "Creative";
        default:                         return "Unknown";
    }
}

// Priority: lower value = higher priority
constexpr uint8_t category_priority(StreamCategory cat) {
    switch (cat) {
        case StreamCategory::Reasoning:  return 0;  // highest
        case StreamCategory::Perception: return 1;
        case StreamCategory::Creative:   return 2;
        case StreamCategory::Memory:     return 3;  // lowest (background)
        default:                         return 255;
    }
}

// Default subsystem flags per category
constexpr Subsystem category_subsystems(StreamCategory cat) {
    switch (cat) {
        case StreamCategory::Perception:
            return Subsystem::Spreading | Subsystem::Salience;
        case StreamCategory::Reasoning:
            return Subsystem::Spreading | Subsystem::Salience | Subsystem::Understanding;
        case StreamCategory::Memory:
            return Subsystem::Salience;  // decay + consolidation
        case StreamCategory::Creative:
            return Subsystem::Curiosity | Subsystem::Understanding;
        default:
            return Subsystem::All;
    }
}

// Default tick interval per category
constexpr std::chrono::milliseconds category_tick_interval(StreamCategory cat) {
    switch (cat) {
        case StreamCategory::Perception: return std::chrono::milliseconds{5};   // fast
        case StreamCategory::Reasoning:  return std::chrono::milliseconds{10};  // normal
        case StreamCategory::Memory:     return std::chrono::milliseconds{50};  // slow
        case StreamCategory::Creative:   return std::chrono::milliseconds{25};  // moderate
        default:                         return std::chrono::milliseconds{10};
    }
}

// Resource budget for a category
struct CategoryBudget {
    uint32_t min_streams = 1;
    uint32_t max_streams = 4;
    uint32_t default_streams = 1;
};

constexpr CategoryBudget category_budget(StreamCategory cat) {
    switch (cat) {
        case StreamCategory::Perception: return {1, 4, 1};
        case StreamCategory::Reasoning:  return {1, 8, 2};
        case StreamCategory::Memory:     return {1, 2, 1};
        case StreamCategory::Creative:   return {1, 4, 1};
        default:                         return {1, 4, 1};
    }
}

// Build a StreamConfig specialized for a given category
inline StreamConfig make_category_config(StreamCategory cat, const StreamConfig& base = {}) {
    StreamConfig cfg = base;
    cfg.subsystem_flags = category_subsystems(cat);
    cfg.tick_interval = category_tick_interval(cat);

    // Memory streams use lighter backoff (they're background)
    if (cat == StreamCategory::Memory) {
        cfg.backoff_strategy = BackoffStrategy::Sleep;
        cfg.sleep_duration = std::chrono::microseconds{2000};
    }
    // Perception needs fastest response
    if (cat == StreamCategory::Perception) {
        cfg.backoff_strategy = BackoffStrategy::SpinYieldSleep;
        cfg.spin_count = 200;
    }

    return cfg;
}

// Complete descriptor for a categorized stream
struct CategorizedStreamInfo {
    uint32_t stream_id = 0;
    StreamCategory category = StreamCategory::Perception;
};

} // namespace brain19
