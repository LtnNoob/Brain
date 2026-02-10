#pragma once
// Lock hierarchy enforcement for Brain19.
// Ensures locks are always acquired in order: LTM → STM → Registry → Embeddings.
// Violations throw in debug builds, are silently ignored in release.

#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

namespace brain19 {

// Lock levels — lower number = must be acquired first.
enum class LockLevel : uint32_t {
    LTM        = 1,
    STM        = 2,
    Registry   = 3,
    Embeddings = 4
};

inline const char* lock_level_name(LockLevel l) {
    switch (l) {
        case LockLevel::LTM:        return "LTM";
        case LockLevel::STM:        return "STM";
        case LockLevel::Registry:   return "Registry";
        case LockLevel::Embeddings: return "Embeddings";
    }
    return "Unknown";
}

// Per-thread tracking of held lock levels (sorted, allows duplicates for re-entrant).
namespace detail {
    inline thread_local std::vector<uint32_t> held_levels;
}

// Check whether acquiring `level` is legal given currently held locks.
// Legal if: no held lock has a strictly higher level than `level`,
// OR `level` equals an already-held level (re-entrant on same level is allowed).
inline void check_hierarchy(LockLevel level) {
#ifdef BRAIN19_DEBUG
    uint32_t val = static_cast<uint32_t>(level);
    if (!detail::held_levels.empty()) {
        uint32_t max_held = *std::max_element(detail::held_levels.begin(),
                                               detail::held_levels.end());
        if (val < max_held) {
            // Violation: trying to acquire a lower-level lock while holding a higher one.
            std::string msg = "Lock hierarchy violation: trying to acquire ";
            msg += lock_level_name(level);
            msg += " (level ";
            msg += std::to_string(val);
            msg += ") while holding level ";
            msg += std::to_string(max_held);
            throw std::runtime_error(msg);
        }
    }
    detail::held_levels.push_back(val);
#else
    (void)level;
#endif
}

inline void release_hierarchy(LockLevel level) {
#ifdef BRAIN19_DEBUG
    uint32_t val = static_cast<uint32_t>(level);
    // Remove one instance (last occurrence) of this level.
    for (auto it = detail::held_levels.rbegin(); it != detail::held_levels.rend(); ++it) {
        if (*it == val) {
            detail::held_levels.erase(std::next(it).base());
            return;
        }
    }
#else
    (void)level;
#endif
}

// Reset thread-local state (useful in tests).
inline void reset_hierarchy() {
#ifdef BRAIN19_DEBUG
    detail::held_levels.clear();
#endif
}

inline const std::vector<uint32_t>& held_lock_levels() {
    return detail::held_levels;
}

// RAII wrapper: wraps a std::shared_mutex and enforces hierarchy on lock/unlock.
class HierarchicalMutex {
public:
    explicit HierarchicalMutex(LockLevel level) : level_(level) {}

    HierarchicalMutex(const HierarchicalMutex&) = delete;
    HierarchicalMutex& operator=(const HierarchicalMutex&) = delete;

    void lock() {
        check_hierarchy(level_);
        mtx_.lock();
    }

    void unlock() {
        mtx_.unlock();
        release_hierarchy(level_);
    }

    void lock_shared() {
        check_hierarchy(level_);
        mtx_.lock_shared();
    }

    void unlock_shared() {
        mtx_.unlock_shared();
        release_hierarchy(level_);
    }

    bool try_lock() {
        check_hierarchy(level_);
        if (mtx_.try_lock()) return true;
        release_hierarchy(level_);
        return false;
    }

    bool try_lock_shared() {
        check_hierarchy(level_);
        if (mtx_.try_lock_shared()) return true;
        release_hierarchy(level_);
        return false;
    }

    LockLevel level() const { return level_; }

private:
    LockLevel level_;
    std::shared_mutex mtx_;
};

// RAII guards
class HierarchicalUniqueLock {
public:
    explicit HierarchicalUniqueLock(HierarchicalMutex& mtx) : mtx_(&mtx) { mtx_->lock(); }
    ~HierarchicalUniqueLock() { if (mtx_) mtx_->unlock(); }
    HierarchicalUniqueLock(const HierarchicalUniqueLock&) = delete;
    HierarchicalUniqueLock& operator=(const HierarchicalUniqueLock&) = delete;
    HierarchicalUniqueLock(HierarchicalUniqueLock&& o) noexcept : mtx_(o.mtx_) { o.mtx_ = nullptr; }
private:
    HierarchicalMutex* mtx_;
};

class HierarchicalSharedLock {
public:
    explicit HierarchicalSharedLock(HierarchicalMutex& mtx) : mtx_(&mtx) { mtx_->lock_shared(); }
    ~HierarchicalSharedLock() { if (mtx_) mtx_->unlock_shared(); }
    HierarchicalSharedLock(const HierarchicalSharedLock&) = delete;
    HierarchicalSharedLock& operator=(const HierarchicalSharedLock&) = delete;
    HierarchicalSharedLock(HierarchicalSharedLock&& o) noexcept : mtx_(o.mtx_) { o.mtx_ = nullptr; }
private:
    HierarchicalMutex* mtx_;
};

} // namespace brain19
