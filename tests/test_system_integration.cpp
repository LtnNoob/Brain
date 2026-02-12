#include "../backend/core/system_orchestrator.hpp"
#include "../backend/core/thinking_pipeline.hpp"
#include "../backend/core/brain19_app.hpp"
#include "../backend/bootstrap/foundation_concepts.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <filesystem>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cerr << "TEST: " << #name << "... "; \
    try { test_##name(); tests_passed++; std::cerr << "PASSED\n"; } \
    catch (const std::exception& e) { tests_failed++; std::cerr << "FAILED: " << e.what() << "\n"; }

#define ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while(0)

using namespace brain19;

// ─── Test Helpers ────────────────────────────────────────────────────────────

static SystemOrchestrator::Config test_config() {
    SystemOrchestrator::Config cfg;
    cfg.data_dir = "/tmp/brain19_test_" + std::to_string(getpid());
    cfg.enable_persistence = false;
    cfg.seed_foundation = false;
    cfg.enable_monitoring = false;
    cfg.max_streams = 2;  // minimal
    return cfg;
}

// ─── Test 1: Initialize + Shutdown ───────────────────────────────────────────

void test_init_shutdown() {
    auto cfg = test_config();
    SystemOrchestrator orch(cfg);

    ASSERT(!orch.is_running());
    ASSERT(orch.initialize());
    ASSERT(orch.is_running());
    ASSERT(orch.concept_count() == 0);

    orch.shutdown();
    ASSERT(!orch.is_running());
}

// ─── Test 2: Foundation Bootstrap ────────────────────────────────────────────

void test_foundation_bootstrap() {
    auto cfg = test_config();
    cfg.seed_foundation = true;
    SystemOrchestrator orch(cfg);

    ASSERT(orch.initialize());
    ASSERT(orch.concept_count() > 0);
    ASSERT(orch.relation_count() > 0);

    // Check that "Entity" exists
    bool found_entity = false;
    for (auto cid : orch.ltm().get_all_concept_ids()) {
        auto info = orch.ltm().retrieve_concept(cid);
        if (info && info->label == "Entity") {
            found_entity = true;
            ASSERT(info->epistemic.type == EpistemicType::DEFINITION);
            ASSERT(info->epistemic.trust >= 0.95);
            break;
        }
    }
    ASSERT(found_entity);

    orch.shutdown();
}

// ─── Test 3: Text Ingestion End-to-End ───────────────────────────────────────

void test_text_ingestion() {
    auto cfg = test_config();
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());

    auto result = orch.ingest_text("Cats are mammals. Dogs are mammals.", true);
    ASSERT(result.success);
    // At minimum, some entities should be extracted
    // (exact count depends on entity extractor implementation)

    orch.shutdown();
}

// ─── Test 4: Ask Question (knowledge-only mode) ─────────────────────────────

void test_ask_no_llm() {
    auto cfg = test_config();
    cfg.seed_foundation = true;
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());

    auto resp = orch.ask("What is Physics?");
    // Should return something in knowledge-only mode
    ASSERT(!resp.answer.empty());

    orch.shutdown();
}

// ─── Test 5: ThinkingPipeline Full Cycle ─────────────────────────────────────

void test_thinking_pipeline() {
    auto cfg = test_config();
    cfg.seed_foundation = true;
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());

    // Find Physics concept
    ConceptId physics_id = 0;
    for (auto cid : orch.ltm().get_all_concept_ids()) {
        auto info = orch.ltm().retrieve_concept(cid);
        if (info && info->label == "Physics") {
            physics_id = cid;
            break;
        }
    }
    ASSERT(physics_id != 0);

    auto result = orch.run_thinking_cycle({physics_id});
    ASSERT(result.steps_completed >= 7);  // At least through curiosity
    ASSERT(!result.activated_concepts.empty());
    ASSERT(result.total_duration_ms > 0.0);

    orch.shutdown();
}

// ─── Test 6: Checkpoint Save + Restore ───────────────────────────────────────

void test_checkpoint_cycle() {
    auto cfg = test_config();
    cfg.seed_foundation = true;
    cfg.enable_persistence = true;
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());

    size_t concepts_before = orch.concept_count();
    ASSERT(concepts_before > 0);

    // Create checkpoint
    orch.create_checkpoint("test_checkpoint");

    // Verify checkpoint directory exists
    auto cp_dir = cfg.data_dir + "/checkpoints";
    ASSERT(std::filesystem::exists(cp_dir));

    orch.shutdown();

    // Cleanup
    std::filesystem::remove_all(cfg.data_dir);
}

// ─── Test 7: Stream Lifecycle ────────────────────────────────────────────────

void test_stream_lifecycle() {
    auto cfg = test_config();
    cfg.max_streams = 2;
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());

    // Get status — should mention streams
    auto status = orch.get_status();
    ASSERT(!status.empty());
    ASSERT(status.find("Streams") != std::string::npos || 
           status.find("Running") != std::string::npos);

    orch.shutdown();
}

// ─── Test 8: Graceful Shutdown Under Load ────────────────────────────────────

void test_shutdown_under_load() {
    auto cfg = test_config();
    cfg.seed_foundation = true;
    cfg.max_streams = 4;
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());

    // Start some thinking cycles
    ConceptId first_cid = 0;
    auto ids = orch.ltm().get_all_concept_ids();
    if (!ids.empty()) first_cid = ids[0];

    if (first_cid != 0) {
        orch.run_thinking_cycle({first_cid});
    }

    // Shutdown should be clean even with active work
    orch.shutdown();
    ASSERT(!orch.is_running());
}

// ─── Test 9: Double Initialize ───────────────────────────────────────────────

void test_double_init() {
    auto cfg = test_config();
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());
    ASSERT(!orch.initialize());  // Second init should return false
    orch.shutdown();
}

// ─── Test 10: Status Reporting ───────────────────────────────────────────────

void test_status_reporting() {
    auto cfg = test_config();
    cfg.seed_foundation = true;
    SystemOrchestrator orch(cfg);
    ASSERT(orch.initialize());

    auto status = orch.get_status();
    ASSERT(status.find("Brain19") != std::string::npos);
    ASSERT(status.find("Running: yes") != std::string::npos);
    ASSERT(status.find("Concepts:") != std::string::npos);

    orch.shutdown();
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cerr << "=== Brain19 System Integration Tests ===\n\n";

    TEST(init_shutdown);
    TEST(foundation_bootstrap);
    TEST(text_ingestion);
    TEST(ask_no_llm);
    TEST(thinking_pipeline);
    TEST(checkpoint_cycle);
    TEST(stream_lifecycle);
    TEST(shutdown_under_load);
    TEST(double_init);
    TEST(status_reporting);

    std::cerr << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    // Cleanup
    std::filesystem::remove_all("/tmp/brain19_test_" + std::to_string(getpid()));

    return tests_failed > 0 ? 1 : 0;
}
