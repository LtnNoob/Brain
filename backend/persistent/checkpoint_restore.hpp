#pragma once
// Phase 4.2: Checkpoint Restore — Load, verify, diff, list checkpoints

#include "checkpoint_manager.hpp"

#include <string>
#include <vector>
#include <set>

namespace brain19 {
namespace persistent {

// Which components to restore
enum class Component : uint8_t {
    LTM        = 0x01,
    STM        = 0x02,
    MODELS     = 0x04,
    KAN        = 0x08,
    COGNITIVE  = 0x10,
    CONFIG     = 0x20,
    ALL        = 0xFF,
};

inline uint8_t operator|(Component a, Component b) { return uint8_t(a) | uint8_t(b); }
inline bool has_component(uint8_t mask, Component c) { return (mask & uint8_t(c)) != 0; }

Component parse_component(const std::string& name);
uint8_t   parse_components(const std::string& csv);

// ─── Restore Results ─────────────────────────────────────────────────────────

struct RestoreResult {
    bool    success = false;
    size_t  concepts_restored  = 0;
    size_t  relations_restored = 0;
    size_t  models_restored    = 0;
    size_t  kan_modules_restored = 0;
    std::string error;
};

// ─── Diff Result ─────────────────────────────────────────────────────────────

struct DiffResult {
    std::string checkpoint_a;
    std::string checkpoint_b;
    
    int64_t concept_count_diff  = 0;
    int64_t relation_count_diff = 0;
    int64_t model_count_diff    = 0;
    int64_t kan_module_diff     = 0;
    int64_t epoch_ms_diff       = 0;
    
    // Files that differ (by hash)
    std::vector<std::string> changed_files;
    
    std::string to_string() const;
};

// ─── Verify Result ───────────────────────────────────────────────────────────

struct VerifyResult {
    bool    valid = false;
    size_t  files_checked = 0;
    size_t  files_ok      = 0;
    std::vector<std::string> failures;
    
    std::string to_string() const;
};

// ─── Checkpoint Restore ──────────────────────────────────────────────────────

class CheckpointRestore {
public:
    // Verify checkpoint integrity (check all SHA-256 hashes)
    static VerifyResult verify(const std::string& checkpoint_dir);
    
    // Load manifest from a checkpoint dir
    static CheckpointManifest load_manifest(const std::string& checkpoint_dir);
    
    // Full or selective restore
    static RestoreResult restore(
        const std::string& checkpoint_dir,
        uint8_t components,
        PersistentLTM*              ltm,
        STMSnapshotData*            stm_out,
        MicroModelRegistry*         models,
        std::vector<std::pair<std::string, KANModule*>>* kan_modules,
        CognitiveState*             cognitive,
        CheckpointConfig*           config
    );
    
    // Diff two checkpoints
    static DiffResult diff(const std::string& dir_a, const std::string& dir_b);
    
    // List all checkpoints with metadata (from a base directory)
    struct ListEntry {
        std::string path;
        std::string timestamp;
        std::string tag;
        uint64_t    epoch_ms;
        uint64_t    concept_count;
        uint64_t    relation_count;
        uint64_t    model_count;
    };
    static std::vector<ListEntry> list(const std::string& base_dir);
    
private:
    // Read helpers
    static bool restore_ltm(const std::string& path, PersistentLTM& ltm);
    static bool restore_stm(const std::string& path, STMSnapshotData& out);
    static bool restore_micromodels(const std::string& path, MicroModelRegistry& reg);
    static bool restore_kan_modules(const std::string& path,
        std::vector<std::pair<std::string, KANModule*>>& modules);
    static bool restore_cognitive(const std::string& path, CognitiveState& out);
    static bool restore_config(const std::string& path, CheckpointConfig& out);
};

} // namespace persistent
} // namespace brain19
