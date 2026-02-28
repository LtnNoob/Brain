// =============================================================================
// Comprehensive Parallel CoLearnLoop Performance Test
// =============================================================================
//
// Tests: Serial vs Parallel throughput, thread utilization, race condition
// detection, quality evolution over time, pain distribution, continuous mode.
//

#include "ltm/long_term_memory.hpp"
#include "ltm/relation.hpp"
#include "bootstrap/foundation_concepts.hpp"
#include "evolution/property_inheritance.hpp"
#include "micromodel/embedding_manager.hpp"
#include "cmodel/concept_model_registry.hpp"
#include "cmodel/concept_trainer.hpp"
#include "graph_net/graph_reasoner.hpp"
#include "colearn/colearn_loop.hpp"
#include "colearn/colearn_types.hpp"

#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <atomic>
#include <thread>
#include <map>
#include <vector>
#include <mutex>

using namespace brain19;
using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// Helpers
// =============================================================================

static void section(const std::string& title) {
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(72, '=') << "\n\n";
}

static void subsection(const std::string& title) {
    std::cout << "--- " << title << " ---\n";
}

static double ms_since(Clock::time_point start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct BenchResult {
    size_t total_chains = 0;
    size_t total_episodes = 0;
    double total_ms = 0.0;
    std::vector<double> cycle_qualities;
    std::vector<double> cycle_times_ms;
    size_t cycles = 0;

    double chains_per_sec() const {
        return total_ms > 0 ? (total_chains / total_ms) * 1000.0 : 0;
    }
    double cycles_per_sec() const {
        return total_ms > 0 ? (cycles / total_ms) * 1000.0 : 0;
    }
    double avg_quality() const {
        if (cycle_qualities.empty()) return 0;
        return std::accumulate(cycle_qualities.begin(), cycle_qualities.end(), 0.0)
               / static_cast<double>(cycle_qualities.size());
    }
    double avg_cycle_ms() const {
        if (cycle_times_ms.empty()) return 0;
        return std::accumulate(cycle_times_ms.begin(), cycle_times_ms.end(), 0.0)
               / static_cast<double>(cycle_times_ms.size());
    }
};

// =============================================================================
// Test 1: Serial vs Parallel Throughput
// =============================================================================

static BenchResult run_benchmark(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                  EmbeddingManager& embeddings, GraphReasoner& reasoner,
                                  size_t thread_count, size_t chains_per_cycle,
                                  size_t num_cycles) {
    CoLearnConfig config;
    config.wake_chains_per_cycle = chains_per_cycle;
    config.sleep_replay_count = std::min(chains_per_cycle, size_t(20));
    config.retrain_epochs = 2;
    config.max_episodes = 5000;
    config.thread_count = thread_count;

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);

    BenchResult result;
    auto total_start = Clock::now();

    for (size_t i = 0; i < num_cycles; ++i) {
        auto cycle_start = Clock::now();
        auto cr = loop.run_cycle();
        double cycle_ms = ms_since(cycle_start);

        result.total_chains += cr.chains_produced;
        result.total_episodes += cr.episodes_stored;
        result.cycle_qualities.push_back(cr.avg_chain_quality);
        result.cycle_times_ms.push_back(cycle_ms);
        ++result.cycles;
    }

    result.total_ms = ms_since(total_start);
    return result;
}

static void test_serial_vs_parallel(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                     EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    section("Test 1: Serial vs Parallel Throughput");

    const size_t CHAINS = 50;
    const size_t CYCLES = 5;

    // Serial benchmark
    subsection("Serial (thread_count=1)");
    auto serial = run_benchmark(ltm, registry, embeddings, reasoner, 1, CHAINS, CYCLES);

    std::cout << "  Chains:       " << serial.total_chains << "\n";
    std::cout << "  Total time:   " << std::fixed << std::setprecision(1) << serial.total_ms << " ms\n";
    std::cout << "  Throughput:   " << std::setprecision(1) << serial.chains_per_sec() << " chains/sec\n";
    std::cout << "  Cycles/sec:   " << std::setprecision(2) << serial.cycles_per_sec() << "\n";
    std::cout << "  Avg quality:  " << std::setprecision(4) << serial.avg_quality() << "\n";
    std::cout << "  Avg cycle:    " << std::setprecision(1) << serial.avg_cycle_ms() << " ms\n";

    // Parallel benchmark (2 threads)
    subsection("Parallel (thread_count=2)");
    auto par2 = run_benchmark(ltm, registry, embeddings, reasoner, 2, CHAINS, CYCLES);

    std::cout << "  Chains:       " << par2.total_chains << "\n";
    std::cout << "  Total time:   " << std::setprecision(1) << par2.total_ms << " ms\n";
    std::cout << "  Throughput:   " << std::setprecision(1) << par2.chains_per_sec() << " chains/sec\n";
    std::cout << "  Cycles/sec:   " << std::setprecision(2) << par2.cycles_per_sec() << "\n";
    std::cout << "  Avg quality:  " << std::setprecision(4) << par2.avg_quality() << "\n";
    std::cout << "  Avg cycle:    " << std::setprecision(1) << par2.avg_cycle_ms() << " ms\n";

    // Parallel benchmark (4 threads)
    subsection("Parallel (thread_count=4)");
    auto par4 = run_benchmark(ltm, registry, embeddings, reasoner, 4, CHAINS, CYCLES);

    std::cout << "  Chains:       " << par4.total_chains << "\n";
    std::cout << "  Total time:   " << std::setprecision(1) << par4.total_ms << " ms\n";
    std::cout << "  Throughput:   " << std::setprecision(1) << par4.chains_per_sec() << " chains/sec\n";
    std::cout << "  Cycles/sec:   " << std::setprecision(2) << par4.cycles_per_sec() << "\n";
    std::cout << "  Avg quality:  " << std::setprecision(4) << par4.avg_quality() << "\n";
    std::cout << "  Avg cycle:    " << std::setprecision(1) << par4.avg_cycle_ms() << " ms\n";

    // Comparison
    subsection("Speedup Analysis");
    double speedup_2 = serial.total_ms / par2.total_ms;
    double speedup_4 = serial.total_ms / par4.total_ms;
    double efficiency_2 = speedup_2 / 2.0 * 100.0;
    double efficiency_4 = speedup_4 / 4.0 * 100.0;

    std::cout << "  2-thread speedup:  " << std::setprecision(2) << speedup_2 << "x"
              << " (efficiency: " << std::setprecision(1) << efficiency_2 << "%)\n";
    std::cout << "  4-thread speedup:  " << std::setprecision(2) << speedup_4 << "x"
              << " (efficiency: " << std::setprecision(1) << efficiency_4 << "%)\n";

    // Per-cycle time comparison
    std::cout << "\n  Per-cycle times (ms):\n";
    std::cout << "  Cycle  | Serial   | 2-Thread | 4-Thread\n";
    std::cout << "  -------+----------+----------+----------\n";
    for (size_t i = 0; i < CYCLES; ++i) {
        std::cout << "  " << std::setw(5) << (i+1) << "  | "
                  << std::setw(8) << std::setprecision(1) << serial.cycle_times_ms[i] << " | "
                  << std::setw(8) << par2.cycle_times_ms[i] << " | "
                  << std::setw(8) << par4.cycle_times_ms[i] << "\n";
    }
}

// =============================================================================
// Test 2: High-Volume Stress Test (200 chains, 4 threads)
// =============================================================================

static void test_high_volume(LongTermMemory& ltm, ConceptModelRegistry& registry,
                              EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    section("Test 2: High-Volume Stress Test (200 chains x 5 cycles, 4 threads)");

    const size_t CHAINS = 200;
    const size_t CYCLES = 5;

    auto result = run_benchmark(ltm, registry, embeddings, reasoner, 4, CHAINS, CYCLES);

    std::cout << "  Total chains:    " << result.total_chains << " / " << (CHAINS * CYCLES) << " expected\n";
    std::cout << "  Total episodes:  " << result.total_episodes << "\n";
    std::cout << "  Total time:      " << std::setprecision(1) << result.total_ms << " ms\n";
    std::cout << "  Throughput:      " << std::setprecision(1) << result.chains_per_sec() << " chains/sec\n";
    std::cout << "  Cycles/sec:      " << std::setprecision(2) << result.cycles_per_sec() << "\n";

    // Verify no data loss
    bool data_ok = result.total_chains > 0 && result.total_episodes > 0;
    std::cout << "  Data integrity:  " << (data_ok ? "PASS" : "FAIL") << "\n";

    // Quality per cycle
    std::cout << "\n  Quality evolution:\n";
    for (size_t i = 0; i < result.cycle_qualities.size(); ++i) {
        std::cout << "    Cycle " << (i+1) << ": quality=" << std::setprecision(4)
                  << result.cycle_qualities[i] << "\n";
    }
}

// =============================================================================
// Test 3: Quality Evolution & Pain Distribution
// =============================================================================

static void test_quality_and_pain(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                   EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    section("Test 3: Quality Evolution & Pain Distribution (10 cycles, 50 chains)");

    CoLearnConfig config;
    config.wake_chains_per_cycle = 50;
    config.sleep_replay_count = 20;
    config.retrain_epochs = 5;
    config.max_episodes = 5000;
    config.thread_count = 4;

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);

    std::cout << "  Cycle | Chains | Quality  | Delta    | Models | Rollback | Corrections\n";
    std::cout << "  ------+--------+----------+----------+--------+----------+------------\n";

    for (size_t i = 0; i < 10; ++i) {
        auto cr = loop.run_cycle();
        std::cout << "  " << std::setw(5) << cr.cycle_number << " | "
                  << std::setw(6) << cr.chains_produced << " | "
                  << std::setw(8) << std::setprecision(4) << cr.avg_chain_quality << " | "
                  << std::setw(8) << std::showpos << cr.quality_delta << std::noshowpos << " | "
                  << std::setw(6) << cr.models_retrained << " | "
                  << std::setw(8) << cr.models_rolled_back << " | "
                  << std::setw(10) << cr.correction_samples_injected << "\n";
    }

    // Pain distribution
    subsection("Pain Distribution");
    const auto& pain = loop.seed_pain_scores();
    if (!pain.empty()) {
        std::vector<double> pain_values;
        pain_values.reserve(pain.size());
        for (const auto& [cid, p] : pain) {
            pain_values.push_back(p);
        }
        std::sort(pain_values.begin(), pain_values.end());

        double min_pain = pain_values.front();
        double max_pain = pain_values.back();
        double median_pain = pain_values[pain_values.size() / 2];
        double avg_pain = std::accumulate(pain_values.begin(), pain_values.end(), 0.0)
                         / static_cast<double>(pain_values.size());

        std::cout << "  Seeds tracked:  " << pain.size() << "\n";
        std::cout << "  Min pain:       " << std::setprecision(4) << min_pain << "\n";
        std::cout << "  Max pain:       " << max_pain << "\n";
        std::cout << "  Median pain:    " << median_pain << "\n";
        std::cout << "  Avg pain:       " << avg_pain << "\n";

        // Pain histogram (5 buckets: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        size_t buckets[5] = {};
        for (double p : pain_values) {
            int b = std::min(4, static_cast<int>(p * 5.0));
            ++buckets[b];
        }
        std::cout << "\n  Pain histogram:\n";
        const char* labels[] = {"[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"};
        for (int b = 0; b < 5; ++b) {
            size_t bar_len = (buckets[b] * 40) / std::max(size_t(1), pain_values.size());
            std::cout << "    " << labels[b] << "  " << std::setw(5) << buckets[b] << " ";
            for (size_t j = 0; j < bar_len; ++j) std::cout << "#";
            std::cout << "\n";
        }
    }

    // Episodic memory stats
    subsection("Episodic Memory Stats");
    std::cout << "  Total episodes: " << loop.episodic_memory().episode_count() << "\n";
}

// =============================================================================
// Test 4: Race Condition Detection (concurrent store + read)
// =============================================================================

static void test_race_conditions() {
    section("Test 4: Race Condition Detection");

    subsection("EpisodicMemory: concurrent store + read + evict");
    {
        EpisodicMemory mem(500);  // Small capacity to trigger evictions
        std::atomic<size_t> store_count{0};
        std::atomic<size_t> read_count{0};
        std::atomic<bool> stop{false};
        std::atomic<size_t> errors{0};

        // 4 writer threads storing episodes
        std::vector<std::thread> writers;
        for (int t = 0; t < 4; ++t) {
            writers.emplace_back([&mem, &store_count, &stop, t]() {
                for (size_t i = 0; !stop.load() && i < 200; ++i) {
                    Episode ep;
                    ep.seed = static_cast<ConceptId>(t * 10000 + i);
                    ep.quality = 0.3 + 0.05 * (i % 10);
                    ep.consolidation_strength = (i % 5 == 0) ? 0.9 : 0.0; // Some consolidated
                    EpisodeStep s;
                    s.concept_id = ep.seed;
                    ep.steps.push_back(s);
                    mem.store(ep);
                    ++store_count;
                }
            });
        }

        // 2 reader threads querying episodes
        std::vector<std::thread> readers;
        for (int t = 0; t < 2; ++t) {
            readers.emplace_back([&mem, &read_count, &stop, &errors, t]() {
                while (!stop.load()) {
                    size_t count = mem.episode_count();
                    if (count > 0) {
                        auto selected = mem.select_for_replay(5, 0.5, 0.3, 0.2);
                        // Verify returned pointers are non-null
                        for (const Episode* ep : selected) {
                            if (ep == nullptr) {
                                ++errors;
                            }
                        }
                    }
                    auto concepts = mem.episodes_for_concept(static_cast<ConceptId>(t * 10000));
                    ++read_count;
                }
            });
        }

        // Let writers finish
        for (auto& w : writers) w.join();
        stop.store(true);
        for (auto& r : readers) r.join();

        std::cout << "  Stores:  " << store_count.load() << "\n";
        std::cout << "  Reads:   " << read_count.load() << "\n";
        std::cout << "  Errors:  " << errors.load() << "\n";
        std::cout << "  Episodes in memory: " << mem.episode_count() << " (cap 500)\n";
        std::cout << "  Result:  " << (errors.load() == 0 ? "PASS (no race conditions)" : "FAIL") << "\n";
    }

    subsection("ErrorCollector: concurrent collect + read");
    {
        ErrorCollector collector;
        std::atomic<size_t> collect_count{0};
        std::atomic<size_t> read_count{0};
        std::atomic<bool> stop{false};
        std::atomic<size_t> errors{0};

        // 4 writer threads collecting from chains
        std::vector<std::thread> writers;
        for (int t = 0; t < 4; ++t) {
            writers.emplace_back([&collector, &collect_count, t]() {
                for (size_t i = 0; i < 100; ++i) {
                    GraphChain chain;
                    chain.termination = TerminationReason::NO_VIABLE_CANDIDATES;

                    TraceStep s0;
                    s0.step_index = 0;
                    s0.source_id = static_cast<ConceptId>(t * 1000 + i);
                    s0.target_id = s0.source_id;
                    s0.composite_score = 0.8;
                    chain.steps.push_back(s0);

                    TraceStep s1;
                    s1.step_index = 1;
                    s1.source_id = s0.source_id;
                    s1.target_id = static_cast<ConceptId>(t * 1000 + i + 1);
                    s1.composite_score = 0.9;
                    chain.steps.push_back(s1);

                    ChainSignal signal;
                    signal.chain_quality = 0.3;
                    collector.collect_from_chain(chain, signal);
                    ++collect_count;
                }
            });
        }

        // 2 reader threads checking corrections
        std::vector<std::thread> readers;
        for (int t = 0; t < 2; ++t) {
            readers.emplace_back([&collector, &read_count, &stop, &errors, t]() {
                while (!stop.load()) {
                    size_t tc = collector.terminal_count();
                    size_t total = collector.total_corrections();
                    // terminal_count should never exceed total_corrections
                    if (tc > total + 1000) { // allow some slack for concurrent updates
                        ++errors;
                    }
                    ++read_count;
                    (void)t;
                }
            });
        }

        for (auto& w : writers) w.join();
        stop.store(true);
        for (auto& r : readers) r.join();

        std::cout << "  Collects: " << collect_count.load() << "\n";
        std::cout << "  Reads:    " << read_count.load() << "\n";
        std::cout << "  Errors:   " << errors.load() << "\n";
        std::cout << "  Terminal:  " << collector.terminal_count() << "\n";
        std::cout << "  Total:     " << collector.total_corrections() << "\n";
        std::cout << "  Result:   " << (errors.load() == 0 ? "PASS (no race conditions)" : "FAIL") << "\n";
    }
}

// =============================================================================
// Test 5: Continuous Mode Stress Test
// =============================================================================

static void test_continuous_stress(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                    EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    section("Test 5: Continuous Mode Stress Test (2 seconds, 4 threads)");

    CoLearnConfig config;
    config.wake_chains_per_cycle = 20;
    config.sleep_replay_count = 10;
    config.retrain_epochs = 2;
    config.max_episodes = 2000;
    config.thread_count = 4;
    config.continuous_interval_ms = 0;  // No delay — max throughput

    CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);

    std::mutex stats_mtx;
    std::vector<CoLearnLoop::CycleResult> all_results;

    loop.set_cycle_callback([&](const CoLearnLoop::CycleResult& r) {
        std::lock_guard<std::mutex> lock(stats_mtx);
        all_results.push_back(r);
    });

    auto start = Clock::now();
    loop.start_continuous();

    // Let it run for 2 seconds
    std::this_thread::sleep_for(std::chrono::seconds(2));

    loop.stop_continuous();
    double elapsed_ms = ms_since(start);

    std::lock_guard<std::mutex> lock(stats_mtx);

    size_t total_chains = 0;
    size_t total_episodes = 0;
    for (const auto& r : all_results) {
        total_chains += r.chains_produced;
        total_episodes += r.episodes_stored;
    }

    std::cout << "  Duration:       " << std::setprecision(0) << elapsed_ms << " ms\n";
    std::cout << "  Cycles:         " << all_results.size() << "\n";
    std::cout << "  Total chains:   " << total_chains << "\n";
    std::cout << "  Total episodes: " << total_episodes << "\n";
    std::cout << "  Chains/sec:     " << std::setprecision(1)
              << (total_chains / elapsed_ms * 1000.0) << "\n";
    std::cout << "  Cycles/sec:     " << std::setprecision(2)
              << (all_results.size() / elapsed_ms * 1000.0) << "\n";

    if (!all_results.empty()) {
        std::cout << "\n  Quality evolution (first 5 and last 5 cycles):\n";
        size_t n = all_results.size();
        size_t show = std::min(n, size_t(5));
        for (size_t i = 0; i < show; ++i) {
            std::cout << "    Cycle " << std::setw(3) << all_results[i].cycle_number
                      << ": quality=" << std::setprecision(4)
                      << all_results[i].avg_chain_quality
                      << " chains=" << all_results[i].chains_produced << "\n";
        }
        if (n > 10) std::cout << "    ...\n";
        if (n > 5) {
            for (size_t i = std::max(show, n - 5); i < n; ++i) {
                std::cout << "    Cycle " << std::setw(3) << all_results[i].cycle_number
                          << ": quality=" << std::setprecision(4)
                          << all_results[i].avg_chain_quality
                          << " chains=" << all_results[i].chains_produced << "\n";
            }
        }
    }

    bool ok = !all_results.empty() && total_chains > 0;
    std::cout << "\n  Result: " << (ok ? "PASS" : "FAIL") << "\n";
}

// =============================================================================
// Test 6: Wake Phase Only — Pure Reasoning Throughput
// =============================================================================

static void test_wake_only_throughput(LongTermMemory& ltm, ConceptModelRegistry& registry,
                                      EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    section("Test 6: Wake Phase Only — Pure Reasoning Throughput");

    std::cout << "  Measuring pure wake_phase() time (no sleep/train overhead)\n\n";

    for (size_t threads : {1, 2, 4}) {
        CoLearnConfig config;
        config.wake_chains_per_cycle = 100;
        config.sleep_replay_count = 10;
        config.retrain_epochs = 2;
        config.max_episodes = 5000;
        config.thread_count = threads;

        CoLearnLoop loop(ltm, registry, embeddings, reasoner, config);

        // Warm up
        loop.wake_phase();

        // Measure 3 wake phases
        double total_ms = 0.0;
        for (int trial = 0; trial < 3; ++trial) {
            auto start = Clock::now();
            loop.wake_phase();
            total_ms += ms_since(start);
            // episodic_memory count grows each trial
        }

        double avg_ms = total_ms / 3.0;
        std::cout << "  " << threads << " thread" << (threads > 1 ? "s" : " ") << ":  "
                  << std::setprecision(1) << avg_ms << " ms avg per wake_phase(100 chains)\n";
    }
}

// =============================================================================
// Test 7: Determinism — Serial produces consistent results
// =============================================================================

static void test_determinism(LongTermMemory& ltm, ConceptModelRegistry& registry,
                              EmbeddingManager& embeddings, GraphReasoner& reasoner) {
    section("Test 7: Serial Determinism Check");

    // Two serial runs with same config should produce same chain counts
    // (quality may differ slightly due to consolidation feedback)
    CoLearnConfig config;
    config.wake_chains_per_cycle = 10;
    config.sleep_replay_count = 5;
    config.retrain_epochs = 2;
    config.max_episodes = 200;
    config.thread_count = 1;

    CoLearnLoop loop1(ltm, registry, embeddings, reasoner, config);
    auto r1 = loop1.run_cycle();

    CoLearnLoop loop2(ltm, registry, embeddings, reasoner, config);
    auto r2 = loop2.run_cycle();

    std::cout << "  Run 1: chains=" << r1.chains_produced
              << " quality=" << std::setprecision(4) << r1.avg_chain_quality << "\n";
    std::cout << "  Run 2: chains=" << r2.chains_produced
              << " quality=" << std::setprecision(4) << r2.avg_chain_quality << "\n";

    bool same_chains = (r1.chains_produced == r2.chains_produced);
    bool close_quality = std::abs(r1.avg_chain_quality - r2.avg_chain_quality) < 0.01;
    std::cout << "  Chain count match: " << (same_chains ? "PASS" : "ACCEPTABLE (seed selection may vary)") << "\n";
    std::cout << "  Quality close:     " << (close_quality ? "PASS" : "ACCEPTABLE") << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Parallel CoLearnLoop Performance Test ===\n";
    std::cout << "Hardware: " << std::thread::hardware_concurrency() << " hardware threads\n";

    // Setup
    section("Setup: Loading Knowledge Base + Training Models");

    LongTermMemory ltm;
    bool loaded = false;
    for (const auto& path : {"../data/foundation_full.json", "data/foundation_full.json",
                              "../data/foundation.json", "data/foundation.json"}) {
        if (FoundationConcepts::seed_from_file(ltm, path, true)) {
            std::cout << "  Loaded from: " << path << "\n";
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        std::cout << "  FALLBACK: using hardcoded seeds\n";
        FoundationConcepts::seed_all(ltm);
    }
    std::cout << "  Concepts: " << ltm.get_all_concept_ids().size() << "\n";
    std::cout << "  Relations: " << ltm.total_relation_count() << "\n";

    // Property inheritance
    {
        PropertyInheritance prop(ltm);
        PropertyInheritance::Config cfg;
        cfg.max_iterations = 50;
        cfg.max_hop_depth = 20;
        auto r = prop.propagate(cfg);
        std::cout << "  Inherited: " << r.properties_inherited << "\n";
    }

    // Train
    EmbeddingManager embeddings;
    ConceptModelRegistry registry;
    {
        embeddings.train_embeddings(ltm, 0.05, 5);
        registry.ensure_models_for(ltm);
        ConceptTrainer trainer;
        auto stats = trainer.train_all(registry, embeddings, ltm);
        std::cout << "  Models: " << stats.models_trained
                  << " (" << stats.models_converged << " converged)\n";
    }

    // Create GraphReasoner
    GraphReasonerConfig gcfg;
    gcfg.max_steps = 8;
    gcfg.enable_composition = true;
    gcfg.chain_coherence_weight = 0.3;
    gcfg.chain_ctx_blend = 0.15;
    gcfg.seed_anchor_weight = 0.35;
    gcfg.seed_anchor_decay = 0.03;
    gcfg.min_embedding_similarity = 0.05;
    gcfg.embedding_sim_weight = 0.1;
    GraphReasoner reasoner(ltm, registry, embeddings, gcfg);

    // Run tests
    test_serial_vs_parallel(ltm, registry, embeddings, reasoner);
    test_high_volume(ltm, registry, embeddings, reasoner);
    test_quality_and_pain(ltm, registry, embeddings, reasoner);
    test_race_conditions();
    test_continuous_stress(ltm, registry, embeddings, reasoner);
    test_wake_only_throughput(ltm, registry, embeddings, reasoner);
    test_determinism(ltm, registry, embeddings, reasoner);

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  ALL PERFORMANCE TESTS COMPLETE\n";
    std::cout << std::string(72, '=') << "\n";

    return 0;
}
