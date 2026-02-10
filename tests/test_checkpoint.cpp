// Phase 4 Tests: Checkpoint Manager
//
// 8 tests covering save, restore, selective restore, rotation,
// integrity, corruption detection, list, and diff.

#include "../backend/persistent/checkpoint_manager.hpp"
#include "../backend/persistent/checkpoint_restore.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <cstring>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;
using namespace brain19;
using namespace brain19::persistent;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    void test_##name(); \
    struct Register_##name { Register_##name() { \
        std::cout << "TEST " #name "... " << std::flush; \
        try { test_##name(); tests_passed++; std::cout << "✓\n"; } \
        catch (const std::exception& e) { tests_failed++; std::cout << "✗ " << e.what() << "\n"; } \
        catch (...) { tests_failed++; std::cout << "✗ unknown exception\n"; } \
    }} reg_##name; \
    void test_##name()

#define ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond " at line " + std::to_string(__LINE__)); } while(0)

static std::string test_dir;

void setup_test_dir() {
    test_dir = "/tmp/brain19_test_checkpoint_" + std::to_string(getpid());
    fs::remove_all(test_dir);
    fs::create_directories(test_dir);
}

void cleanup_test_dir() {
    fs::remove_all(test_dir);
}

// Helper: create a PersistentLTM with some data
std::unique_ptr<PersistentLTM> make_test_ltm(const std::string& dir) {
    fs::create_directories(dir);
    auto ltm = std::make_unique<PersistentLTM>(dir);
    
    EpistemicMetadata meta(EpistemicType::FACT, EpistemicStatus::ACTIVE, 0.9);
    ltm->store_concept("gravity", "Force of attraction between masses", meta);
    ltm->store_concept("mass", "Amount of matter in an object", meta);
    ltm->store_concept("energy", "Capacity to do work", meta);
    
    ltm->add_relation(1, 2, RelationType::CAUSES, 0.8);
    ltm->add_relation(2, 3, RelationType::SIMILAR_TO, 0.6);
    
    return ltm;
}

// ─── Test 1: Save full checkpoint ────────────────────────────────────────────

TEST(save_full_checkpoint) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    STMSnapshotData stm;
    stm.timestamp = 1234567890;
    SnapshotContext ctx;
    ctx.context_id = 1;
    SnapshotConcept sc; sc.concept_id = 1; sc.activation = 0.9; sc.classification = ActivationClass::CORE_KNOWLEDGE;
    ctx.concepts.push_back(sc);
    stm.contexts.push_back(ctx);
    
    CognitiveState cog;
    cog.focus_set = {1, 2};
    cog.avg_activation = 0.75;
    cog.tick_count = 42;
    
    CheckpointConfig config;
    config.entries["learning_rate"] = "0.01";
    config.entries["max_concepts"] = "10000";
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    opts.tag = "test";
    CheckpointManager mgr(opts);
    
    auto path = mgr.save(ltm.get(), &stm, nullptr, nullptr, &cog, &config);
    
    ASSERT(fs::exists(path));
    ASSERT(fs::exists(path + "/manifest.json"));
    ASSERT(fs::exists(path + "/ltm.bin"));
    ASSERT(fs::exists(path + "/stm.bin"));
    ASSERT(fs::exists(path + "/cognitive.bin"));
    ASSERT(fs::exists(path + "/config.json"));
    
    cleanup_test_dir();
}

// ─── Test 2: Restore full checkpoint + verify state ──────────────────────────

TEST(restore_full_checkpoint) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    STMSnapshotData stm;
    stm.timestamp = 9999;
    SnapshotContext ctx;
    ctx.context_id = 42;
    SnapshotConcept sc2; sc2.concept_id = 1; sc2.activation = 0.5; sc2.classification = ActivationClass::CONTEXTUAL;
    ctx.concepts.push_back(sc2);
    stm.contexts.push_back(ctx);
    
    CognitiveState cog;
    cog.focus_set = {1, 3};
    cog.avg_activation = 0.6;
    cog.tick_count = 100;
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    CheckpointManager mgr(opts);
    auto path = mgr.save(ltm.get(), &stm, nullptr, nullptr, &cog, nullptr);
    
    // Restore into fresh objects
    STMSnapshotData stm_restored;
    CognitiveState cog_restored;
    
    auto result = CheckpointRestore::restore(
        path, uint8_t(Component::ALL),
        nullptr, &stm_restored, nullptr, nullptr, &cog_restored, nullptr
    );
    
    ASSERT(result.success);
    ASSERT(stm_restored.timestamp == 9999);
    ASSERT(stm_restored.contexts.size() == 1);
    ASSERT(stm_restored.contexts[0].context_id == 42);
    ASSERT(cog_restored.focus_set.size() == 2);
    ASSERT(cog_restored.tick_count == 100);
    
    cleanup_test_dir();
}

// ─── Test 3: Selective restore (only LTM) ────────────────────────────────────

TEST(selective_restore_ltm) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    STMSnapshotData stm;
    stm.timestamp = 5555;
    
    CognitiveState cog;
    cog.tick_count = 200;
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    CheckpointManager mgr(opts);
    auto path = mgr.save(ltm.get(), &stm, nullptr, nullptr, &cog, nullptr);
    
    // Restore only STM (not LTM, not cognitive)
    STMSnapshotData stm_restored;
    CognitiveState cog_restored;
    
    auto result = CheckpointRestore::restore(
        path, uint8_t(Component::STM),
        nullptr, &stm_restored, nullptr, nullptr, &cog_restored, nullptr
    );
    
    ASSERT(result.success);
    ASSERT(stm_restored.timestamp == 5555);
    // Cognitive should NOT have been restored (not in component mask)
    ASSERT(cog_restored.tick_count == 0);
    
    cleanup_test_dir();
}

// ─── Test 4: Checkpoint rotation ─────────────────────────────────────────────

TEST(checkpoint_rotation) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    opts.max_keep = 3;
    
    // Create 5 checkpoints
    for (int i = 0; i < 5; ++i) {
        opts.tag = "v" + std::to_string(i);
        CheckpointManager mgr(opts);
        mgr.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
        // Sleep briefly so timestamps differ
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Should only have 3 left
    auto dirs = CheckpointRestore::list(test_dir + "/checkpoints");
    ASSERT(dirs.size() == 3);
    
    cleanup_test_dir();
}

// ─── Test 5: Integrity verification (happy path) ─────────────────────────────

TEST(integrity_verify_ok) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    CheckpointManager mgr(opts);
    auto path = mgr.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
    
    auto vr = CheckpointRestore::verify(path);
    ASSERT(vr.valid);
    ASSERT(vr.files_checked > 0);
    ASSERT(vr.files_ok == vr.files_checked);
    
    cleanup_test_dir();
}

// ─── Test 6: Corrupted checkpoint detection ──────────────────────────────────

TEST(corrupted_checkpoint_detection) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    CheckpointManager mgr(opts);
    auto path = mgr.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
    
    // Tamper with ltm.bin
    std::string ltm_path = path + "/ltm.bin";
    {
        std::ofstream f(ltm_path, std::ios::binary | std::ios::app);
        f.write("CORRUPTED", 9);
    }
    
    auto vr = CheckpointRestore::verify(path);
    ASSERT(!vr.valid);
    ASSERT(!vr.failures.empty());
    
    // Restore should also fail
    auto result = CheckpointRestore::restore(
        path, uint8_t(Component::ALL),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
    );
    ASSERT(!result.success);
    
    cleanup_test_dir();
}

// ─── Test 7: List checkpoints with metadata ──────────────────────────────────

TEST(list_checkpoints) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    std::string base = test_dir + "/checkpoints";
    
    for (int i = 0; i < 3; ++i) {
        CheckpointManager::Options opts;
        opts.base_dir = base;
        opts.tag = "tag" + std::to_string(i);
        CheckpointManager mgr(opts);
        mgr.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    auto entries = CheckpointRestore::list(base);
    ASSERT(entries.size() == 3);
    
    // Should be sorted newest first
    ASSERT(entries[0].epoch_ms >= entries[1].epoch_ms);
    ASSERT(entries[1].epoch_ms >= entries[2].epoch_ms);
    
    // All should have 3 concepts
    for (auto& e : entries) {
        ASSERT(e.concept_count == 3);
        ASSERT(e.relation_count == 2);
    }
    
    cleanup_test_dir();
}

// ─── Test 8: Diff two checkpoints ───────────────────────────────────────────

TEST(diff_checkpoints) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    std::string base = test_dir + "/checkpoints";
    CheckpointManager::Options opts;
    opts.base_dir = base;
    opts.max_keep = 10;
    
    opts.tag = "before";
    CheckpointManager mgr1(opts);
    auto path_a = mgr1.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Add more concepts
    EpistemicMetadata meta2(EpistemicType::THEORY, EpistemicStatus::ACTIVE, 0.95);
    ltm->store_concept("time", "Temporal dimension", meta2);
    ltm->store_concept("space", "Spatial dimension", meta2);
    ltm->add_relation(3, 4, RelationType::SIMILAR_TO, 0.7);
    
    opts.tag = "after";
    CheckpointManager mgr2(opts);
    auto path_b = mgr2.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
    
    auto diff = CheckpointRestore::diff(path_a, path_b);
    ASSERT(diff.concept_count_diff == 2);  // 5 - 3
    ASSERT(diff.relation_count_diff == 1); // 3 - 2
    ASSERT(diff.epoch_ms_diff > 0);
    ASSERT(!diff.changed_files.empty());  // ltm.bin should differ
    
    cleanup_test_dir();
}

// ─── Test 9: SHA-256 correctness ─────────────────────────────────────────────

TEST(sha256_correctness) {
    // Test vector: SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    std::string input = "abc";
    auto hash = SHA256::hash_bytes(
        reinterpret_cast<const uint8_t*>(input.data()), input.size());
    ASSERT(hash == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    
    // Empty string: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    auto empty_hash = SHA256::hash_bytes(nullptr, 0);
    ASSERT(empty_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

// ─── Test 10: Corrupt manifest JSON ──────────────────────────────────────────

TEST(corrupt_manifest_json) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    opts.tag = "test";
    CheckpointManager mgr(opts);
    
    auto path = mgr.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
    ASSERT(!path.empty());
    
    // Corrupt the manifest.json
    {
        std::ofstream f(path + "/manifest.json", std::ios::trunc);
        f << "{ this is not valid json !!!";
    }
    
    auto vr = CheckpointRestore::verify(path);
    ASSERT(!vr.valid);
    
    cleanup_test_dir();
}

// ─── Test 11: Truncated manifest (empty file) ───────────────────────────────

TEST(truncated_manifest) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    opts.tag = "test";
    CheckpointManager mgr(opts);
    
    auto path = mgr.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
    ASSERT(!path.empty());
    
    // Truncate manifest to empty
    { std::ofstream f(path + "/manifest.json", std::ios::trunc); }
    
    auto vr = CheckpointRestore::verify(path);
    ASSERT(!vr.valid);
    
    cleanup_test_dir();
}

// ─── Test 12: Missing manifest file ─────────────────────────────────────────

TEST(missing_manifest) {
    setup_test_dir();
    
    std::string ltm_dir = test_dir + "/ltm_data";
    auto ltm = make_test_ltm(ltm_dir);
    
    CheckpointManager::Options opts;
    opts.base_dir = test_dir + "/checkpoints";
    opts.tag = "test";
    CheckpointManager mgr(opts);
    
    auto path = mgr.save(ltm.get(), nullptr, nullptr, nullptr, nullptr, nullptr);
    ASSERT(!path.empty());
    
    // Delete manifest
    fs::remove(path + "/manifest.json");
    
    auto vr = CheckpointRestore::verify(path);
    ASSERT(!vr.valid);
    
    cleanup_test_dir();
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Brain19 Checkpoint Tests ===\n\n";
    
    // Tests are auto-registered by static constructors above
    
    std::cout << "\n" << tests_passed << " passed, " << tests_failed << " failed\n";
    return tests_failed > 0 ? 1 : 0;
}
