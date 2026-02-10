#pragma once
// Lightweight deadlock detection for Brain19 debug builds.
// Logs lock acquisitions per thread and optionally detects cycles in the
// wait-for graph.  Active only when BRAIN19_DEBUG is defined.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brain19 {

#ifdef BRAIN19_DEBUG

struct LockEvent {
    std::thread::id thread;
    uint32_t level;         // LockLevel as uint32_t
    const char* label;
    enum Action : uint8_t { Acquire, Release } action;
};

class DeadlockDetector {
public:
    static DeadlockDetector& instance() {
        static DeadlockDetector dd;
        return dd;
    }

    static constexpr size_t max_log_size = 10000;

    void on_acquire(std::thread::id tid, uint32_t level, const char* label) {
        std::lock_guard lk(mtx_);
        if (log_.size() >= max_log_size) {
            // Rotate: drop oldest half
            log_.erase(log_.begin(), log_.begin() + static_cast<ptrdiff_t>(max_log_size / 2));
        }
        log_.push_back({tid, level, label, LockEvent::Acquire});
        held_[tid].insert(level);
    }

    void on_release(std::thread::id tid, uint32_t level, const char* label) {
        std::lock_guard lk(mtx_);
        if (log_.size() >= max_log_size) {
            log_.erase(log_.begin(), log_.begin() + static_cast<ptrdiff_t>(max_log_size / 2));
        }
        log_.push_back({tid, level, label, LockEvent::Release});
        held_[tid].erase(level);
    }

    // Record that `tid` is waiting to acquire `level`.
    void on_wait(std::thread::id tid, uint32_t level) {
        std::lock_guard lk(mtx_);
        waiting_[tid] = level;
    }

    void on_wait_done(std::thread::id tid) {
        std::lock_guard lk(mtx_);
        waiting_.erase(tid);
    }

    // Simple cycle detection in wait-for graph.
    // Returns true if a potential deadlock is found.
    bool has_cycle() const {
        std::lock_guard lk(mtx_);
        // Build: for each waiting thread, find threads that hold the waited-for level.
        // Then check if any of those threads are waiting for something held by
        // the first thread (or transitively).
        for (auto& [waiting_tid, wanted_level] : waiting_) {
            std::unordered_set<std::thread::id, ThreadIdHash> visited;
            if (dfs_cycle(waiting_tid, visited)) return true;
        }
        return false;
    }

    // Get full acquisition log (for debugging).
    std::vector<LockEvent> get_log() const {
        std::lock_guard lk(mtx_);
        return log_;
    }

    void clear() {
        std::lock_guard lk(mtx_);
        log_.clear();
        held_.clear();
        waiting_.clear();
    }

    size_t log_size() const {
        std::lock_guard lk(mtx_);
        return log_.size();
    }

private:
    DeadlockDetector() = default;

    struct ThreadIdHash {
        size_t operator()(std::thread::id id) const {
            return std::hash<std::thread::id>{}(id);
        }
    };

    bool dfs_cycle(std::thread::id tid,
                   std::unordered_set<std::thread::id, ThreadIdHash>& visited) const {
        if (!visited.insert(tid).second) return true; // cycle!
        auto wit = waiting_.find(tid);
        if (wit == waiting_.end()) return false;
        uint32_t wanted = wit->second;
        // Find all threads that hold `wanted`.
        for (auto& [other_tid, levels] : held_) {
            if (other_tid == tid) continue;
            if (levels.count(wanted)) {
                if (dfs_cycle(other_tid, visited)) return true;
            }
        }
        return false;
    }

    mutable std::mutex mtx_;
    std::vector<LockEvent> log_;
    std::unordered_map<std::thread::id, std::unordered_set<uint32_t>, ThreadIdHash> held_;
    std::unordered_map<std::thread::id, uint32_t, ThreadIdHash> waiting_;
};

#else

// No-op in release builds.
class DeadlockDetector {
public:
    static DeadlockDetector& instance() { static DeadlockDetector dd; return dd; }
    void on_acquire(std::thread::id, uint32_t, const char*) {}
    void on_release(std::thread::id, uint32_t, const char*) {}
    void on_wait(std::thread::id, uint32_t) {}
    void on_wait_done(std::thread::id) {}
    bool has_cycle() const { return false; }
    size_t log_size() const { return 0; }
    void clear() {}
};

#endif

} // namespace brain19
