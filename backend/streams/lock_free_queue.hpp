#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <type_traits>
#include <vector>

namespace brain19 {

// =============================================================================
// Vyukov bounded MPMC queue — ABA-safe via sequence counters
// =============================================================================
template <typename T>
class MPMCQueue {
    static_assert(std::is_nothrow_move_constructible_v<T> || std::is_nothrow_copy_constructible_v<T>,
                  "T must be nothrow move- or copy-constructible");
public:
    explicit MPMCQueue(size_t capacity)
        : capacity_(next_pow2(std::max(capacity, size_t(2))))
        , mask_(capacity_ - 1)
        , buffer_(capacity_)
        , head_(0)
        , tail_(0)
    {
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    // Try to enqueue. Returns false if full.
    bool try_push(const T& value) {
        Cell* cell;
        size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false; // full
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
        cell->data = value;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool try_push(T&& value) {
        Cell* cell;
        size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false;
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
        cell->data = std::move(value);
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Try to dequeue. Returns nullopt if empty.
    std::optional<T> try_pop() {
        Cell* cell;
        size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return std::nullopt; // empty
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
        T result = std::move(cell->data);
        cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
        return result;
    }

    size_t capacity() const { return capacity_; }

    bool empty() const {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_relaxed);
        return h >= t;
    }

    // Approximate size (may be stale, suitable for load-balancing heuristics)
    size_t size_approx() const {
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t h = head_.load(std::memory_order_relaxed);
        return t > h ? t - h : 0;
    }

private:
    struct alignas(64) Cell {
        std::atomic<size_t> sequence;
        T data{};
    };

    static size_t next_pow2(size_t v) {
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16; v |= v >> 32;
        return v + 1;
    }

    const size_t capacity_;
    const size_t mask_;
    std::vector<Cell> buffer_;

    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

// =============================================================================
// SPSC bounded queue — simpler, no CAS needed
// =============================================================================
template <typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(size_t capacity)
        : capacity_(next_pow2(std::max(capacity, size_t(2))))
        , mask_(capacity_ - 1)
        , buffer_(capacity_)
        , head_(0)
        , tail_(0)
    {}

    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    bool try_push(const T& value) {
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t next = (t + 1) & mask_;
        if (next == head_.load(std::memory_order_acquire)) return false;
        buffer_[t] = value;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    bool try_push(T&& value) {
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t next = (t + 1) & mask_;
        if (next == head_.load(std::memory_order_acquire)) return false;
        buffer_[t] = std::move(value);
        tail_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> try_pop() {
        size_t h = head_.load(std::memory_order_relaxed);
        if (h == tail_.load(std::memory_order_acquire)) return std::nullopt;
        T result = std::move(buffer_[h]);
        head_.store((h + 1) & mask_, std::memory_order_release);
        return result;
    }

    bool empty() const {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

    size_t capacity() const { return capacity_ - 1; } // one slot reserved

private:
    static size_t next_pow2(size_t v) {
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16; v |= v >> 32;
        return v + 1;
    }

    const size_t capacity_;
    const size_t mask_;
    std::vector<T> buffer_;

    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

} // namespace brain19
