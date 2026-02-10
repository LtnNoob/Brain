// test_lock_hierarchy.cpp — Lock hierarchy enforcement & deadlock detection tests
// Build: make test_lock_hierarchy
// Must be compiled with -DBRAIN19_DEBUG to enable hierarchy checks.

#ifndef BRAIN19_DEBUG
#define BRAIN19_DEBUG
#endif

#include "../backend/concurrent/lock_hierarchy.hpp"
#include "../backend/concurrent/deadlock_detector.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace brain19;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        ++tests_run; \
        reset_hierarchy(); \
        DeadlockDetector::instance().clear(); \
        std::cout << "  [" << tests_run << "] " << name << " ... "; \
        std::cout.flush(); \
    } while(0)

#define PASS() \
    do { ++tests_passed; std::cout << "PASS\n"; } while(0)

#define FAIL(msg) \
    do { std::cout << "FAIL: " << msg << "\n"; } while(0)

// ==========================================================================
// Test 1: Correct lock order LTM → STM → Registry → Embeddings
// ==========================================================================
void test_correct_order() {
    TEST("Correct lock order (LTM→STM→Registry→Embeddings)");

    HierarchicalMutex ltm_mtx(LockLevel::LTM);
    HierarchicalMutex stm_mtx(LockLevel::STM);
    HierarchicalMutex reg_mtx(LockLevel::Registry);
    HierarchicalMutex emb_mtx(LockLevel::Embeddings);

    try {
        HierarchicalUniqueLock l1(ltm_mtx);
        HierarchicalSharedLock l2(stm_mtx);
        HierarchicalUniqueLock l3(reg_mtx);
        HierarchicalSharedLock l4(emb_mtx);
        // All four held — no exception.
        assert(held_lock_levels().size() == 4);
        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}

// ==========================================================================
// Test 2: Wrong order detected (STM before LTM)
// ==========================================================================
void test_wrong_order_detected() {
    TEST("Wrong order detected (STM before LTM)");

    HierarchicalMutex ltm_mtx(LockLevel::LTM);
    HierarchicalMutex stm_mtx(LockLevel::STM);

    bool caught = false;
    try {
        HierarchicalUniqueLock l1(stm_mtx);  // STM first
        HierarchicalUniqueLock l2(ltm_mtx);  // LTM second — violation!
        FAIL("No exception thrown");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find("hierarchy violation") != std::string::npos) {
            caught = true;
        }
    }
    if (caught) {
        PASS();
    } else {
        FAIL("Expected hierarchy violation exception");
    }
}

// ==========================================================================
// Test 3: Multi-thread correct hierarchy
// ==========================================================================
void test_multithread_correct() {
    TEST("Multi-thread correct hierarchy");

    HierarchicalMutex ltm_mtx(LockLevel::LTM);
    HierarchicalMutex stm_mtx(LockLevel::STM);
    HierarchicalMutex reg_mtx(LockLevel::Registry);

    constexpr int N = 4;
    std::latch start_latch(N);
    std::atomic<int> errors{0};

    auto worker = [&]() {
        reset_hierarchy(); // each thread needs clean thread-local state
        start_latch.arrive_and_wait();
        try {
            for (int i = 0; i < 100; ++i) {
                HierarchicalUniqueLock l1(ltm_mtx);
                HierarchicalSharedLock l2(stm_mtx);
                HierarchicalUniqueLock l3(reg_mtx);
                // small work
            }
        } catch (...) {
            errors.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    if (errors.load() == 0) {
        PASS();
    } else {
        FAIL("Unexpected exceptions in worker threads");
    }
}

// ==========================================================================
// Test 4: Re-entrant lock on same level — allowed
// ==========================================================================
void test_reentrant_same_level() {
    TEST("Re-entrant lock on same level — allowed");

    HierarchicalMutex mtx_a(LockLevel::Registry);
    HierarchicalMutex mtx_b(LockLevel::Registry);  // different mutex, same level

    try {
        HierarchicalUniqueLock l1(mtx_a);
        HierarchicalUniqueLock l2(mtx_b);  // same level — should be allowed
        assert(held_lock_levels().size() == 2);
        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}

// ==========================================================================
// Test 5: Stress test — 8 threads, all four lock levels
// ==========================================================================
void test_stress_8_threads() {
    TEST("Stress test: 8 threads × 4 lock levels");

    HierarchicalMutex ltm_mtx(LockLevel::LTM);
    HierarchicalMutex stm_mtx(LockLevel::STM);
    HierarchicalMutex reg_mtx(LockLevel::Registry);
    HierarchicalMutex emb_mtx(LockLevel::Embeddings);

    constexpr int N = 8;
    constexpr int ITERS = 200;
    std::latch start_latch(N);
    std::atomic<int> errors{0};
    std::atomic<uint64_t> total_ops{0};

    auto worker = [&]() {
        reset_hierarchy();
        start_latch.arrive_and_wait();
        try {
            for (int i = 0; i < ITERS; ++i) {
                // Correct order: LTM → STM → Registry → Embeddings
                {
                    HierarchicalSharedLock l1(ltm_mtx);
                    HierarchicalSharedLock l2(stm_mtx);
                }
                {
                    HierarchicalUniqueLock l1(reg_mtx);
                    HierarchicalSharedLock l2(emb_mtx);
                }
                {
                    HierarchicalSharedLock l1(ltm_mtx);
                    HierarchicalSharedLock l2(stm_mtx);
                    HierarchicalSharedLock l3(reg_mtx);
                    HierarchicalSharedLock l4(emb_mtx);
                }
                total_ops.fetch_add(3, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            errors.fetch_add(1);
            std::cerr << "  Stress error: " << e.what() << "\n";
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    if (errors.load() == 0) {
        std::cout << "PASS (" << total_ops.load() << " lock ops)\n";
        ++tests_passed;
    } else {
        FAIL(std::to_string(errors.load()) + " thread(s) failed");
    }
}

// ==========================================================================
// Test 6: Simulated ThinkStream lock pattern under load
// ==========================================================================
void test_think_stream_pattern() {
    TEST("ThinkStream lock pattern under load");

    // ThinkStream accesses: LTM (read) → STM (write) in do_spreading,
    // and LTM (read) alone in do_curiosity. We simulate this pattern.

    HierarchicalMutex ltm_mtx(LockLevel::LTM);
    HierarchicalMutex stm_mtx(LockLevel::STM);
    HierarchicalMutex reg_mtx(LockLevel::Registry);
    HierarchicalMutex emb_mtx(LockLevel::Embeddings);

    constexpr int STREAMS = 8;
    constexpr int TICKS = 100;
    std::latch start_latch(STREAMS);
    std::atomic<int> errors{0};
    std::atomic<uint64_t> total_ticks{0};

    auto simulate_stream = [&]() {
        reset_hierarchy();
        start_latch.arrive_and_wait();
        try {
            for (int tick = 0; tick < TICKS; ++tick) {
                // do_spreading: read LTM, then write STM
                {
                    HierarchicalSharedLock ltm_lock(ltm_mtx);
                    // read concept relations...
                    HierarchicalUniqueLock stm_lock(stm_mtx);
                    // boost concepts in STM...
                }

                // do_salience: only STM
                {
                    HierarchicalUniqueLock stm_lock(stm_mtx);
                    // decay...
                }

                // do_curiosity: read LTM, write STM
                {
                    HierarchicalSharedLock ltm_lock(ltm_mtx);
                    HierarchicalUniqueLock stm_lock(stm_mtx);
                }

                // do_understanding: read LTM, read/write STM, optionally read embeddings
                {
                    HierarchicalSharedLock ltm_lock(ltm_mtx);
                    HierarchicalUniqueLock stm_lock(stm_mtx);
                    HierarchicalSharedLock emb_lock(emb_mtx);
                }

                total_ticks.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            errors.fetch_add(1);
            std::cerr << "  Stream error: " << e.what() << "\n";
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < STREAMS; ++i) threads.emplace_back(simulate_stream);
    for (auto& t : threads) t.join();

    if (errors.load() == 0) {
        std::cout << "PASS (" << total_ticks.load() << " ticks across "
                  << STREAMS << " streams)\n";
        ++tests_passed;
    } else {
        FAIL(std::to_string(errors.load()) + " stream(s) failed");
    }
}

// ==========================================================================
// Test 7: DeadlockDetector logging
// ==========================================================================
void test_deadlock_detector_logging() {
    TEST("DeadlockDetector logs acquisitions");

    auto& dd = DeadlockDetector::instance();
    auto tid = std::this_thread::get_id();

    dd.on_acquire(tid, 1, "LTM");
    dd.on_acquire(tid, 2, "STM");
    dd.on_release(tid, 2, "STM");
    dd.on_release(tid, 1, "LTM");

    if (dd.log_size() == 4) {
        PASS();
    } else {
        FAIL("Expected 4 log entries, got " + std::to_string(dd.log_size()));
    }
}

// ==========================================================================
// Test 8: Wrong order — Registry before STM
// ==========================================================================
void test_wrong_order_registry_before_stm() {
    TEST("Wrong order: Embeddings held, then LTM — violation");

    HierarchicalMutex ltm_mtx(LockLevel::LTM);
    HierarchicalMutex emb_mtx(LockLevel::Embeddings);

    bool caught = false;
    try {
        HierarchicalUniqueLock l1(emb_mtx);  // Embeddings (level 4)
        HierarchicalUniqueLock l2(ltm_mtx);  // LTM (level 1) — violation!
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find("hierarchy violation") != std::string::npos) {
            caught = true;
        }
    }
    if (caught) {
        PASS();
    } else {
        FAIL("Expected hierarchy violation exception");
    }
}

// ==========================================================================
int main() {
    std::cout << "=== Brain19 Lock Hierarchy Tests ===\n\n";

    test_correct_order();
    test_wrong_order_detected();
    test_multithread_correct();
    test_reentrant_same_level();
    test_stress_8_threads();
    test_think_stream_pattern();
    test_deadlock_detector_logging();
    test_wrong_order_registry_before_stm();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run
              << " passed ===\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
